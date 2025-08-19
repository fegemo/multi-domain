import logging
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility import io_utils, keras_utils, palette_utils
from utility.keras_utils import ZeroCenteredGradientPenalty, NoopGradientPenalty, LinearAnnealingScheduler, \
    NoopAnnealingScheduler
from .side2side_model import S2SModel
from .networks import munit_content_encoder, munit_style_encoder, munit_decoder, munit_discriminator_multi_scale


class MunitModel(S2SModel):
    def __init__(self, config, export_additional_training_endpoint=False):
        self.discriminators = None
        self.generators = None
        self.decoders = None
        self.style_encoders = None
        self.content_encoders = None
        self.gen_supplier = keras_utils.NParamsSupplier(3 if config.palette_quantization else 2)
        super().__init__(config, export_additional_training_endpoint)
        self.lambda_adversarial = config.lambda_adversarial
        self.lambda_reconstruction = config.lambda_l1
        self.lambda_latent_reconstruction = config.lambda_latent_reconstruction
        self.lambda_cyclic_reconstruction = config.lambda_cyclic_reconstruction
        self.lambda_regularization = config.lambda_regularization
        if config.adv == "lsgan":
            self.adv_loss = keras_utils.LSGANLoss(config.number_of_domains, config.discriminator_scales, True)
        elif config.adv == "r3gan":
            self.adv_loss = keras_utils.RelativisticLoss(config.number_of_domains, config.discriminator_scales, True)
        else:
            raise ValueError(
                f"The provided {config.adv} type of adversarial loss has not been implemented for {type(self)}")
        if config.lambda_gp > 0:
            self.gradient_penalty = ZeroCenteredGradientPenalty()
        else:
            self.gradient_penalty = NoopGradientPenalty()
        if config.palette_quantization and config.annealing != "none":
            self.annealing_scheduler = LinearAnnealingScheduler(config.temperature, [d.quantization for d in self.decoders])
        else:
            self.annealing_scheduler = NoopAnnealingScheduler()

    def create_inference_networks(self):
        config = self.config
        image_size = config.image_size
        inner_channels = config.inner_channels
        palette_quantization = config.palette_quantization
        temperature = config.temperature
        domain_letters = [name[0].upper() for name in config.domains]
        if config.generator in ["munit", ""]:
            content_encoders = [munit_content_encoder(s, image_size, inner_channels) for s in domain_letters]
            style_encoders = [munit_style_encoder(s, image_size, inner_channels) for s in domain_letters]
            decoders = [munit_decoder(s, inner_channels, palette_quantization, temperature) for s in domain_letters]

            input_layer = tf.keras.layers.Input(shape=(config.image_size, config.image_size, config.output_channels))
            palette_input_layer = tf.keras.layers.Input(shape=(None, config.output_channels))
            x = input_layer
            generators = [tf.keras.Model(x, decoders[i](
                self.gen_supplier(style_encoders[i](x), content_encoders[i](x), palette_input_layer)))
                          for i in range(config.number_of_domains)]

            self.content_encoders = content_encoders
            self.style_encoders = style_encoders
            self.decoders = decoders
            # generators is a "virtual" model that joins a corresponding content and style encoder with a decoder
            # it is an auto-encoder that can be used to generate images from and to a given domain
            # it is also used to update the networks' weights
            self.generators = generators

            return {
                "content_encoders": content_encoders,
                "style_encoders": style_encoders,
                "decoders": decoders
            }
        else:
            raise ValueError(f"The provided {config.generator} type of generator has not been implemented")

    def create_training_only_networks(self):
        config = self.config
        image_size = config.image_size
        inner_channels = config.inner_channels
        scales = config.discriminator_scales
        domain_letters = [name[0].upper() for name in config.domains]
        if config.generator in ["munit", ""]:
            discriminators = [munit_discriminator_multi_scale(s, image_size, inner_channels, scales)
                              for s in domain_letters]
            self.discriminators = discriminators
            return {
                "discriminators": discriminators
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type of discriminator has not been implemented")

    def generator_loss(self, predicted_patches_fake, predicted_patches_real, reconstructed_images,
                       original_images, recoded_style, original_random_style, recoded_content, original_content,
                       cyclic_reconstruction):
        number_of_domains = self.config.number_of_domains
        discriminator_scales = self.config.discriminator_scales
        # loss from the discriminator
        adversarial_loss = self.adv_loss.calculate_generator_loss(predicted_patches_fake, predicted_patches_real)
        # adversarial_loss (shape=[d])

        # within domain reconstruction (auto-encoder L1 regression)
        same_domain_reconstruction = [tf.reduce_mean(tf.abs(original_images[i] - reconstructed_images[i]))
                                      for i in range(number_of_domains)]
        # same_domain_reconstruction (shape=[d])

        # style reconstruction
        style_reconstruction = [tf.reduce_mean(tf.abs(original_random_style[i] - recoded_style[i]))
                                for i in range(number_of_domains)]
        # style_reconstruction (shape=[d])

        # content reconstruction
        content_reconstruction = [tf.reduce_mean(tf.abs(original_content[i] - recoded_content[i]))
                                  for i in range(number_of_domains)]
        # content_reconstruction (shape=[d])

        # cross domain reconstruction (translation) - this is not used for edges/shoes or bags, but for landscapes
        cross_domain_reconstruction = [tf.reduce_mean(tf.abs(original_images[i] - cyclic_reconstruction[i]))
                                       for i in range(number_of_domains)]

        # l2 regularization
        l2_regularization = [tf.reduce_sum(self.decoders[i].losses) for i in range(number_of_domains)]

        total_loss = [self.lambda_adversarial * adversarial_loss[i] +
                      self.lambda_reconstruction * same_domain_reconstruction[i] +
                      self.lambda_latent_reconstruction * style_reconstruction[i] +
                      self.lambda_latent_reconstruction * content_reconstruction[i] +
                      self.lambda_cyclic_reconstruction * cross_domain_reconstruction[i] +
                      self.lambda_regularization * l2_regularization[i]
                      for i in range(number_of_domains)]

        return {"adversarial": adversarial_loss, "same-domain": same_domain_reconstruction,
                "style": style_reconstruction, "content": content_reconstruction,
                "cross-domain": cross_domain_reconstruction,
                "l2-regularization": l2_regularization,
                "total": total_loss}

    def discriminator_loss(self, predicted_patches_real, predicted_patches_fake, r1_penalty, r2_penalty):
        number_of_domains = self.config.number_of_domains
        # discriminator_scales = self.config.discriminator_scales

        # shape=[d, ds] x [b, x, x, 1] => [d, ds] => [d]
        adversarial_loss, real_loss, fake_loss = self.adv_loss.calculate_discriminator_loss(
            predicted_patches_fake, predicted_patches_real)

        # l2 regularization
        l2_regularization = [tf.reduce_sum(self.discriminators[i].losses) for i in range(number_of_domains)]

        # r1/r2 regularization
        gp_loss = [(r1_penalty[d] + r2_penalty[d])  / 2. for d in range(number_of_domains)]
        # gp_loss (shape=[d])

        total_loss = [adversarial_loss[i] +
            self.lambda_regularization * l2_regularization[i] +
            self.config.lambda_gp * gp_loss[i]
            for i in range(number_of_domains)]

        return {
            "adversarial": adversarial_loss,
            "real": real_loss,
            "fake": fake_loss,
            "l2-regularization": l2_regularization,
            "gradient-penalty": gp_loss,
            "total": total_loss
        }

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        """
        From the pytorch reference implementation:
        https://github.com/NVlabs/MUNIT/blob/master/train.py#L58
        dis_update():
        - gets random style codes
        - encodes the real images and get only the content code
        - decodes, translating images to another domain, using content from real images and random styles
        - computes loss with the fake image translated and the real (both in the same domain)
          - tf.reduce_mean((fake_patches-0)**2 + tf.reduce_mean((real_patches-1)**2)

        gen_update():
        - gets random style codes
        - encodes the real images and get both content and style codes
        - decode to reconstruct within the same domain
        - decode translating to another domain using random style code
        - encode the translated images
        - decode the translated images to their original domain using the original style code (unused)
        - computes loss:
          - within domain image reconstruction (10)
          - style reconstruction (1)
          - content reconstruction (1)
          - cyclic reconstruction (0, unused)
          - gan loss: tf.reduce_mean((fake_patches-1)**2) (1)

        :param batch:
        :param step:
        :param evaluate_steps:
        :param t:
        :return:
        """
        temperature = self.annealing_scheduler.update(t)

        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = self.config.number_of_domains, batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # a. find a random source domain for each domain
        random_shift = tf.random.uniform((), minval=1, maxval=number_of_domains, dtype=tf.int32)
        random_source_domain = tf.roll(tf.range(number_of_domains, dtype=tf.int32), shift=random_shift, axis=0)
        # random_shift (shape=[])
        # random_source_domain (shape=[d])
        # b. find a random style code for each image in the batch
        random_style_codes = [tf.random.normal([batch_size, 8], mean=0.0, stddev=1.0)
                              for _ in range(number_of_domains)]
        # random_style_codes (list of d * shape=[b, 8])

        # c. extract the palettes of the input images
        palette = [palette_utils.batch_extract_palette_ragged(batch[i]) for i in range(number_of_domains)]
        # palette (d x shape=[b, (n), c])

        with tf.GradientTape(persistent=True) as tape:
            # 1. encode the input images
            encoded_contents = [self.content_encoders[i](batch[i], training=True) for i in range(number_of_domains)]
            encoded_styles = [self.style_encoders[i](batch[i], training=True) for i in range(number_of_domains)]
            # encoded_contents (list of d * shape=[b, 16, 16, 256])
            # encoded_styles (list of d * shape=[b, 8])
            # encoded_contents_a = self.content_encoders[0](batch[0])
            # encoded_contents_b = self.content_encoders[1](batch[1])
            # encoded_contents_c = self.content_encoders[2](batch[2])
            # encoded_contents_d = self.content_encoders[3](batch[3])
            # encoded_contents = tf.stack([encoded_contents_a, encoded_contents_b, encoded_contents_c, encoded_contents_d], axis=0)

            # encoded_styles_a = self.style_encoders[0](batch[0])
            # encoded_styles_b = self.style_encoders[1](batch[1])
            # encoded_styles_c = self.style_encoders[2](batch[2])
            # encoded_styles_d = self.style_encoders[3](batch[3])
            # encoded_styles = tf.stack([encoded_styles_a, encoded_styles_b, encoded_styles_c, encoded_styles_d], axis=0)

            # 2. decode the encoded input images (within the same domain) to perfectly reconstruct them
            decoded_images = [
                self.decoders[i](
                    self.gen_supplier(encoded_styles[i], encoded_contents[i], palette[i]), training=True
                )["output_image"]
                for i in range(number_of_domains)]
            # decoded_images (list of d * shape=[b, s, s, c])
            # decoded_images_a = self.decoders[0]([encoded_styles_a, encoded_contents_a])[0]
            # decoded_images_b = self.decoders[1]([encoded_styles_b, encoded_contents_b])[0]
            # decoded_images_c = self.decoders[2]([encoded_styles_c, encoded_contents_c])[0]
            # decoded_images_d = self.decoders[3]([encoded_styles_d, encoded_contents_d])[0]
            # decoded_images = tf.stack([decoded_images_a, decoded_images_b, decoded_images_c, decoded_images_d], axis=0)

            # 3. decode the encoded input images (to a random target domain and using a different style code)
            translated_images = [self.decoders[i](
                self.gen_supplier(random_style_codes[i], tf.gather(encoded_contents, random_source_domain[i]),
                                  palette[i]), training=True)["output_image"]
                                 for i in range(number_of_domains)]
            # translated_images (list of d * shape=[b, s, s, c])
            # translated_images_a = self.decoders[0]([random_style_codes[0], tf.gather(encoded_contents, random_source_domain[0])])[0]
            # translated_images_b = self.decoders[1]([random_style_codes[1], tf.gather(encoded_contents, random_source_domain[1])])[0]
            # translated_images_c = self.decoders[2]([random_style_codes[2], tf.gather(encoded_contents, random_source_domain[2])])[0]
            # translated_images_d = self.decoders[3]([random_style_codes[3], tf.gather(encoded_contents, random_source_domain[3])])[0]
            # translated_images = tf.stack([translated_images_a, translated_images_b, translated_images_c, translated_images_d], axis=0)

            # 4. encode again the cross-domain translated images
            # we need to rearrange the encoded_translated_contents, as they are the reconstruction of the content in the
            # random source domain. We do not need to rearrange the encoded_styles
            encoded_translated_contents = [self.content_encoders[i](translated_images[i], training=True)
                                           for i in range(number_of_domains)]
            encoded_translated_contents = tf.stack(encoded_translated_contents)
            encoded_translated_contents = tf.gather(encoded_translated_contents, random_source_domain)
            # encoded_translated_contents (shape=[d, b, 16, 16, 256])
            encoded_translated_styles = [self.style_encoders[i](translated_images[i], training=True)
                                         for i in range(number_of_domains)]
            # encoded_translated_styles (list of d * shape=[b, 8])

            # 5. decode once more (used only for winter<>summer and cityscapes<>synthia datasets in munit)
            decoded_translated_images = [self.decoders[i](self.gen_supplier(encoded_styles[i],
                                                                            encoded_translated_contents[i],
                                                                            palette[i]), training=True)["output_image"]
                                         for i in range(number_of_domains)]

            # 6. discriminate the input images (real, then fake from cross-domain translation)
            predicted_patches_real = [self.discriminators[i](batch[i], training=True) for i in range(number_of_domains)]
            predicted_patches_fake = [self.discriminators[i](translated_images[i], training=True) for i in range(number_of_domains)]

            r1_penalty = self.gradient_penalty(self.discriminators, batch)
            r2_penalty = self.gradient_penalty(self.discriminators, translated_images)

            d_loss = self.discriminator_loss(predicted_patches_real, predicted_patches_fake, r1_penalty, r2_penalty)
            g_loss = self.generator_loss(predicted_patches_fake, predicted_patches_real,
                                         decoded_images, batch, encoded_translated_styles,
                                         random_style_codes, encoded_translated_contents, encoded_contents,
                                         decoded_translated_images)

        # 7. apply the gradients to the models
        # since the update to keras 3.0 (due to tensorflow 2.18), the optimizer needs to be called with
        # only a single set of gradients and variables. Therefore, we need to concatenate the gradients and
        # variables of all models before calling the optimizer
        discriminator_gradients = [tape.gradient(d_loss["total"][d], self.discriminators[d].trainable_variables)
                                   for d in range(number_of_domains)]
        discriminator_gradients = [g for grad in discriminator_gradients for g in grad]
        generator_gradients = [tape.gradient(g_loss["total"][d], self.generators[d].trainable_variables)
                               for d in range(number_of_domains)]
        generator_gradients = [g for grad in generator_gradients for g in grad]

        all_discriminator_trainable_variables = [v for d in range(number_of_domains)
                                                 for v in self.discriminators[d].trainable_variables]
        all_generator_trainable_variables = [v for d in range(number_of_domains)
                                             for v in self.generators[d].trainable_variables]
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, all_discriminator_trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, all_generator_trainable_variables))

        # write statistics of the training step
        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                adv_loss = fake_loss = real_loss = weight_loss = gp_loss = total_loss = 0
                for i in range(number_of_domains):
                    adv_loss += d_loss["adversarial"][i]
                    fake_loss += d_loss["fake"][i]
                    real_loss += d_loss["real"][i]
                    weight_loss += d_loss["l2-regularization"][i]
                    gp_loss += d_loss["gradient-penalty"][i]
                    total_loss += d_loss["total"][i]
                tf.summary.scalar("adversarial_loss", tf.reduce_mean(adv_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("fake_loss", tf.reduce_mean(fake_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("real_loss", tf.reduce_mean(real_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("weight_loss", tf.reduce_mean(weight_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("gradient_penalty_loss", tf.reduce_mean(gp_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss / number_of_domains),
                                  step=step // evaluate_steps)

        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                adversarial_loss = same_domain_loss = style_loss = content_loss = cross_domain_loss = weight_loss = total_loss = 0
                for i in range(number_of_domains):
                    adversarial_loss += g_loss["adversarial"][i]
                    same_domain_loss += g_loss["same-domain"][i]
                    style_loss += g_loss["style"][i]
                    content_loss += g_loss["content"][i]
                    cross_domain_loss += g_loss["cross-domain"][i]
                    weight_loss += g_loss["l2-regularization"][i]
                    total_loss += g_loss["total"][i]
                tf.summary.scalar("adversarial_loss", tf.reduce_mean(adversarial_loss), step=step // evaluate_steps)
                tf.summary.scalar("same_domain_loss", tf.reduce_mean(same_domain_loss), step=step // evaluate_steps)
                tf.summary.scalar("style_loss", tf.reduce_mean(style_loss), step=step // evaluate_steps)
                tf.summary.scalar("content_loss", tf.reduce_mean(content_loss), step=step // evaluate_steps)
                tf.summary.scalar("cross_domain_loss", tf.reduce_mean(cross_domain_loss),
                                  step=step // evaluate_steps)
                tf.summary.scalar("weight_loss", tf.reduce_mean(weight_loss), step=step // evaluate_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss), step=step // evaluate_steps)

    def select_examples_for_visualization(self, train_ds, test_ds):
        number_of_domains = self.config.number_of_domains
        number_of_examples = 3
        ensure_inside_range = lambda x: x % number_of_domains
        train_examples = []
        test_examples = []

        train_ds_iter = train_ds.unbatch().take(number_of_examples).as_numpy_iterator()
        test_ds_iter = test_ds.shuffle(self.config.test_size).unbatch().take(number_of_examples).as_numpy_iterator()
        for c in range(number_of_examples):
            source_index = ensure_inside_range(c + 0)
            target_index = ensure_inside_range(c + 1)

            train_domain_images = next(train_ds_iter)
            random_style_code = tf.random.normal([1, 8], mean=0.0, stddev=1.0)
            train_examples.append(
                (train_domain_images[source_index], train_domain_images[target_index],
                 random_style_code, source_index, target_index))

            test_domain_images = next(test_ds_iter)
            random_style_code = tf.random.normal([1, 8], mean=0.0, stddev=1.0)
            test_examples.append(
                (test_domain_images[source_index], test_domain_images[target_index],
                 random_style_code, source_index, target_index)
            )

        return train_examples + test_examples

    def preview_generated_images_during_training(self, examples, save_name, step):
        title = ["Input", "Reconstruc.", "Rnd. Style", "Translated", "Target", "Cyclic"]
        num_rows = len(examples)
        num_cols = len(title)

        if step is not None:
            if step == 1:
                step = 0
            title[1] += f" ({step / 1000}k)"
            title[2] += f" ({step / 1000}k)"
            title[3] += f" ({step / 1000}k)"
            title[5] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i in range(num_rows):
            source_image = examples[i][0]
            target_image = examples[i][1]
            random_target_style = examples[i][2]
            source_domain = examples[i][3]
            target_domain = examples[i][4]
            palette = palette_utils.batch_extract_palette_ragged(source_image[tf.newaxis, ...])
            source_image_content = self.content_encoders[source_domain](source_image[tf.newaxis, ...])
            source_image_style = self.style_encoders[source_domain](source_image[tf.newaxis, ...])
            for j in range(num_cols):
                index = i * num_cols + j + 1
                plt.subplot(num_rows, num_cols, index)
                plt.title(title[j], fontdict={"fontsize": 24}, wrap=True)
                image = None
                if j == 0:
                    image = source_image
                elif j == 1:
                    # Reconstructed (VAE)
                    output = self.decoders[source_domain](
                        self.gen_supplier(source_image_style, source_image_content, palette)
                    )
                    image = output["output_image"]
                elif j == 2:
                    # Random Style, same domain
                    output = self.decoders[source_domain](
                        self.gen_supplier(random_target_style, source_image_content, palette)
                    )
                    image = output["output_image"]
                elif j == 3:
                    # Translated, different domain
                    output = self.decoders[target_domain](
                        self.gen_supplier(random_target_style, source_image_content, palette)
                    )
                    image = output["output_image"]
                elif j == 4:
                    # Target
                    image = target_image
                elif j == 5:
                    # Cyclic, same domain
                    output = self.decoders[target_domain](
                        self.gen_supplier(random_target_style, source_image_content, palette)
                    )
                    translated_image = output["output_image"]
                    content = self.content_encoders[target_domain](translated_image)
                    output = self.decoders[source_domain](
                        self.gen_supplier(source_image_style, content, palette)
                    )
                    image = output["output_image"]
                image = tf.squeeze(image)
                plt.imshow((image + 1.) / 2.)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)
        return figure

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        number_of_domains = self.config.number_of_domains

        def initialize_random_examples_from_dataset(dataset):
            domain_images = next(iter(dataset.unbatch().take(num_images)))

            random_source_indices = tf.random.uniform([num_images], minval=0, maxval=number_of_domains, dtype="int32")
            random_target_indices = tf.random.uniform([num_images], minval=0, maxval=number_of_domains, dtype="int32")

            source_images = tf.gather(domain_images, random_source_indices)
            target_images = tf.gather(domain_images, random_target_indices)

            return target_images, source_images, random_target_indices, random_source_indices

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):

        def generate_images_from_dataset(dataset_name):
            target_images, source_images, target_domains, source_domains = example_indices_for_evaluation[dataset_name]
            number_of_examples = len(source_images)
            fake_images = np.empty((number_of_examples, self.config.image_size, self.config.image_size,
                                    self.config.output_channels), dtype=np.float32)
            batch_size = self.config.batch
            for batch_start in range(0, number_of_examples, batch_size):
                batch_end = min(batch_start + batch_size, number_of_examples)

                source_images_slice = source_images[batch_start:batch_end]
                source_domains_slice = source_domains[batch_start:batch_end]
                target_domains_slice = target_domains[batch_start:batch_end]
                palette_slice = palette_utils.batch_extract_palette_ragged(source_images_slice)

                for i in range(batch_end - batch_start):
                    decoder = self.decoders[target_domains_slice[i]]
                    style_encoder = self.style_encoders[target_domains_slice[i]]
                    content_encoder = self.content_encoders[source_domains_slice[i]]
                    output_slice = decoder(
                        self.gen_supplier(style_encoder(tf.expand_dims(source_images_slice[i], 0)),
                                          content_encoder(tf.expand_dims(source_images_slice[i], 0)),
                                          tf.expand_dims(palette_slice[i], 0))
                    )
                    fake_images_slice = output_slice["output_image"]
                    fake_images[batch_start + i] = fake_images_slice

            logging.debug(
                f"Generated all {number_of_examples} fake images for {dataset_name} dataset, which occupy {fake_images.nbytes / 1024 / 1024} MB.")
            return target_images, tf.constant(fake_images)

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        """
        Generates two figures as a grid of images, one of translations to other domains with the original style code
        and another with translations to other domains with random style codes.
        Each row has a different source domain from { back, left, front, right } with the real input image and the
        translated images in the other columns.

        Layout of the first figure:
        (same style code)
         back  fleft  ffront  fright
        fback   left  ffront  fright
        fback  fleft   front  fright
        fback  fleft  ffront   right

        Layout of the second figure:
        (rnd style code)
         back  fleft  ffront  fright
        fback   left  ffront  fright
        fback  fleft   front  fright
        fback  fleft  ffront   right

        :param enumerated_dataset:
        :param step:
        :param num_images:
        :return:
        """
        base_image_path = self.get_output_folder("test-images")
        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains

        # dataset = list(dataset.take(num_images).as_numpy_iterator())
        # list of b * shape=[d, s, s, c]

        titles = [*self.config.domains]
        num_cols = number_of_domains
        num_rows = number_of_domains

        # for each image in the dataset...
        example_idx = 0
        # for domain_images in dataset:
        for c, domain_images in tqdm(enumerated_dataset, total=num_images):
            images_same_style = []
            images_rnd_style = []
            for i in range(number_of_domains):
                source_domain = i
                source_image = domain_images[source_domain]
                source_content = self.content_encoders[source_domain](source_image[tf.newaxis, ...])
                source_style = self.style_encoders[source_domain](source_image[tf.newaxis, ...])
                palette = palette_utils.batch_extract_palette_ragged(source_image[tf.newaxis, ...])

                images_same_style.append([])
                images_rnd_style.append([])
                for j in range(number_of_domains):
                    target_domain = j
                    if target_domain == source_domain:
                        images_same_style[i].append(source_image)
                        images_rnd_style[i].append(source_image)
                        continue
                    else:
                        random_style_code = tf.random.normal([1, 8], mean=0.0, stddev=1.0)
                        translated_image = self.decoders[target_domain](
                            self.gen_supplier(source_style, source_content, palette)
                        )["output_image"]
                        translated_image_rnd = self.decoders[target_domain](
                            self.gen_supplier(random_style_code, source_content, palette)
                        )["output_image"]
                        images_same_style[i].append(tf.squeeze(translated_image))
                        images_rnd_style[i].append(tf.squeeze(translated_image_rnd))

            fig_images = (images_same_style, images_rnd_style)
            fig_names = ("Same Style Code", "Random Style Code")
            for images, fig_name in zip(fig_images, fig_names):
                fig_title = fig_name.replace(" ", "_").lower()
                fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
                for i in range(num_rows):
                    for j in range(num_cols):
                        title = titles[j] if i != j else "Input"
                        plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
                        plt.title(title, fontdict={"fontsize": 20})
                        plt.imshow(images[i][j] * 0.5 + 0.5)
                        plt.axis("off")
                fig.tight_layout()

                image_path = os.sep.join([base_image_path, f"{example_idx}_at_step_{step}_{fig_title}.png"])
                plt.savefig(image_path, transparent=True)
                plt.close(fig)
            example_idx += 1

        print(f"Generated {num_images * 2} images in the test-images folder.")

    def debug_discriminator_output(self, batch, image_path):
        discriminator_scales = self.config.discriminator_scales
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch (shape=(d, b, s, s, c))
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size = batch_shape[0], batch_shape[1], batch_shape[2]
        domain_images = tf.transpose(batch, [1, 0, 2, 3, 4])
        # domain_images (shape=[b, d, s, s, c])

        ensure_inside_range = lambda x: tf.math.floormod(x, number_of_domains)
        source_domains = tf.range(number_of_domains)[:batch_size]
        target_domains = ensure_inside_range(source_domains + 1)

        real_images = tf.gather(domain_images, source_domains, batch_dims=1)
        # real_images (shape=[b, s, s, c])
        palette = palette_utils.batch_extract_palette_ragged(real_images)
        fake_images = []
        for i in range(batch_size):
            content_code = self.content_encoders[source_domains[i]](real_images[i][tf.newaxis, ...])
            random_style_code = tf.random.normal([1, 8], mean=0.0, stddev=1.0)
            fake_image = self.decoders[target_domains[i]](
                self.gen_supplier(random_style_code, content_code, palette[i][tf.newaxis, ...]), training=True
            )["output_image"]
            fake_images.append(fake_image)
        fake_images = tf.concat(fake_images, axis=0)
        # fake_images (shape=[b, s, s, c])

        # gets the result of discriminating the real and fake (translated) images
        real_patches = [self.discriminators[source_domains[i]](real_images[i][tf.newaxis, ...])
                        for i in range(batch_size)]
        fake_patches = [self.discriminators[target_domains[i]](fake_images[i][tf.newaxis, ...])
                        for i in range(batch_size)]
        # if discriminator_scales == 1:
        #     real_patches = [[real_patches[i]] for i in range(batch_size)]
        #     fake_patches = [[fake_patches[i]] for i in range(batch_size)]
        # [b] x [ds] x shape=[1, x, x, 1]

        real_means = [tf.reduce_mean(real_patches[i][c], axis=[1, 2, 3]) for i in range(batch_size)
                      for c in range(discriminator_scales)]
        fake_means = [tf.reduce_mean(fake_patches[i][c], axis=[1, 2, 3]) for i in range(batch_size)
                      for c in range(discriminator_scales)]
        # real_means = tf.reshape(real_means, [batch_size, 3])
        # fake_means = tf.reshape(fake_means, [batch_size, 3])
        # [b] x [3] x shape=[1]

        # lsgan yields an unbounded real number, which should be 1 for real images and 0 for fake
        # but, we need to provide them in the [0, 1] range
        flattened_real_patches = tf.concat([tf.squeeze(tf.reshape(real_patches[i][c], [-1])) for i in range(batch_size)
                                            for c in range(discriminator_scales)], axis=0)
        flattened_fake_patches = tf.concat([tf.squeeze(tf.reshape(fake_patches[i][c], [-1])) for i in range(batch_size)
                                            for c in range(discriminator_scales)], axis=0)
        concatenated_predictions = tf.concat([flattened_real_patches, flattened_fake_patches], axis=0)
        min_value = tf.reduce_min(concatenated_predictions)
        max_value = tf.reduce_max(concatenated_predictions)
        amplitude = max_value - min_value
        real_predicted = [[(real_patches[i][c] - min_value) / amplitude for c in range(discriminator_scales)]
                          for i in range(batch_size)]
        fake_predicted = [[(fake_patches[i][c] - min_value) / amplitude for c in range(discriminator_scales)]
                          for i in range(batch_size)]

        discriminator_titles = [f"Disc. Scale {c}" for c in range(discriminator_scales)]
        titles = ["Real", *discriminator_titles, "Translated", *discriminator_titles]
        num_cols = len(titles)
        num_rows = batch_size.numpy()

        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows, num_cols, (i * num_cols) + j + 1)
                subplot_title = ""
                if i == 0:
                    subplot_title = titles[j]
                plt.title(subplot_title, fontdict={"fontsize": 20})

                imshow_args = {}
                if j == 0:
                    image = real_images[i] * 0.5 + 0.5
                elif 0 < j < discriminator_scales + 1:
                    image = tf.squeeze(real_predicted[i][j - 1])
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                elif j == discriminator_scales + 1:
                    image = fake_images[i] * 0.5 + 0.5
                elif j > discriminator_scales + 1:
                    image = tf.squeeze(fake_predicted[i][j - discriminator_scales - 2])
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                else:
                    raise ValueError(f"Invalid column index {j}")
                plt.axis("off")
                plt.imshow(image, **imshow_args)

        fig.tight_layout()
        plt.savefig(image_path, transparent=True)
        plt.close(fig)
