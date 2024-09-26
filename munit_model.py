import logging

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from side2side_model import S2SModel
from networks import munit_content_encoder, munit_style_encoder, munit_decoder, munit_discriminator_multi_scale


class MunitModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_reconstruction = config.lambda_l1
        self.lambda_latent_reconstruction = config.lambda_latent_reconstruction

    def create_inference_networks(self):
        config = self.config
        domain_letters = [name[0].upper() for name in config.domains]
        if config.generator in ["munit", ""]:
            content_encoders = [munit_content_encoder(s) for s in domain_letters]
            style_encoders = [munit_style_encoder(s) for s in domain_letters]
            decoders = [munit_decoder(s) for s in domain_letters]

            input_layer = tf.keras.layers.Input(shape=(config.image_size, config.image_size, config.output_channels))
            x = input_layer
            generators = [tf.keras.Model(x, decoders[i]([style_encoders[i](x), content_encoders[i](x)]))
                          for i in range(config.number_of_domains)]

            self.content_encoders = content_encoders
            self.style_encoders = style_encoders
            self.decoders = decoders
            # generators is a "virtual" model that joins a corresponding content and style encoder with a decoder
            # it is an auto-encoder that can be used to generate images from and to a given domain
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
        domain_letters = [name[0].upper() for name in config.domains]
        if config.generator in ["munit", ""]:
            discriminators = [munit_discriminator_multi_scale(s) for s in domain_letters]
            self.discriminators = discriminators
            return {
                "discriminators": discriminators
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type of discriminator has not been implemented")

    def generator_loss(self, predicted_patches_fake, reconstructed_images,
                       original_images, recoded_style, original_random_style, recoded_content, original_content):
        number_of_domains = self.config.number_of_domains
        # loss from the discriminator
        adversarial_loss = [
            tf.reduce_mean(tf.square(predicted_patches_fake[d][out] - tf.ones_like(predicted_patches_fake[d][out])))
            for out in range(3)
            for d in range(number_of_domains)]
        adversarial_loss = tf.reshape(adversarial_loss, [number_of_domains, 3])
        adversarial_loss = tf.reduce_mean(adversarial_loss, axis=1)
        # within domain reconstruction (auto-encoder L1 regression)
        same_domain_reconstruction = [tf.reduce_mean(tf.abs(original_images[i] - reconstructed_images[i]))
                                      for i in range(number_of_domains)]
        # style reconstruction
        style_reconstruction = [tf.reduce_mean(tf.abs(original_random_style[i] - recoded_style[i]))
                                for i in range(number_of_domains)]
        # content reconstruction
        content_reconstruction = [tf.reduce_mean(tf.abs(original_content[i] - recoded_content[i]))
                                  for i in range(number_of_domains)]
        # cross domain reconstruction (translation) - this is not used in the tf implementation of munit
        # cross_domain_reconstruction = [tf.reduce_mean(tf.abs(original_images[i] - cyclic_reconstruction[i]))
        #                                for i in range(number_of_domains)]

        # l2 regularization
        l2_regularization = [tf.reduce_sum(self.decoders[i].losses) for i in range(number_of_domains)]

        total_loss = [adversarial_loss[i] +
                      self.lambda_reconstruction * same_domain_reconstruction[i] +
                      self.lambda_latent_reconstruction * style_reconstruction[i] +
                      self.lambda_latent_reconstruction * content_reconstruction[i]
                      # self.lambda_cyclic_reconstruction * cross_domain_reconstruction[i]
                      + l2_regularization[i]
                      for i in range(number_of_domains)]

        return {"adversarial": adversarial_loss, "same-domain": same_domain_reconstruction,
                "style": style_reconstruction, "content": content_reconstruction,
                "l2-regularization": l2_regularization,
                "total": total_loss}

    def discriminator_loss(self, predicted_patches_real, predicted_patches_fake):
        number_of_domains = self.config.number_of_domains

        # shape=[d, 3] x [b, x, x, 1] => [d, 3] => [d]
        real_loss = [
            tf.reduce_mean(tf.square(predicted_patches_real[d][out] - tf.ones_like(predicted_patches_real[d][out])))
            for out in range(3)
            for d in range(number_of_domains)]
        real_loss = tf.reshape(real_loss, [number_of_domains, 3])
        real_loss = tf.reduce_mean(real_loss, axis=1)

        fake_loss = [
            tf.reduce_mean(tf.square(predicted_patches_fake[d][out] - tf.zeros_like(predicted_patches_fake[d][out])))
            for out in range(3)
            for d in range(number_of_domains)]
        fake_loss = tf.reshape(fake_loss, [number_of_domains, 3])
        fake_loss = tf.reduce_mean(fake_loss, axis=1)

        # l2 regularization
        l2_regularization = [tf.reduce_sum(self.discriminators[i].losses) for i in range(number_of_domains)]

        total_loss = [real_loss[i] + fake_loss[i] + l2_regularization[i] for i in range(number_of_domains)]

        return {"real": real_loss, "fake": fake_loss, "l2-regularization": l2_regularization, "total": total_loss}

    @tf.function
    def train_step(self, batch, step, update_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = self.config.number_of_domains, batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # a. find a random source domain for each domain
        random_shift = tf.random.uniform((), minval=1, maxval=number_of_domains, dtype=tf.int32)
        random_source_domain = tf.roll(tf.range(number_of_domains, dtype=tf.int32), shift=random_shift, axis=0)
        # b. find a random style code for each image in the batch
        random_style_codes = [tf.random.normal([batch_size, 8], mean=0.0, stddev=1.0)
                              for _ in range(self.config.number_of_domains)]
        with tf.GradientTape(persistent=True) as tape:
            # 1. encode the input images
            encoded_contents = [self.content_encoders[i](batch[i]) for i in range(number_of_domains)]
            encoded_styles = [self.style_encoders[i](batch[i]) for i in range(number_of_domains)]
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

            # 2. decode the encoded input images (within the same domain)
            decoded_images = [self.decoders[i]([encoded_styles[i], encoded_contents[i]])[0]
                              for i in range(number_of_domains)]
            # decoded_images_a = self.decoders[0]([encoded_styles_a, encoded_contents_a])[0]
            # decoded_images_b = self.decoders[1]([encoded_styles_b, encoded_contents_b])[0]
            # decoded_images_c = self.decoders[2]([encoded_styles_c, encoded_contents_c])[0]
            # decoded_images_d = self.decoders[3]([encoded_styles_d, encoded_contents_d])[0]
            # decoded_images = tf.stack([decoded_images_a, decoded_images_b, decoded_images_c, decoded_images_d], axis=0)

            # 3. decode the encoded input images (to a random target domain and using a different style code)
            translated_images = [self.decoders[i](
                [random_style_codes[i], tf.gather(encoded_contents, tf.gather(random_source_domain, i))])[0]
                                 for i in range(number_of_domains)]
            # translated_images_a = self.decoders[0]([random_style_codes[0], tf.gather(encoded_contents, random_source_domain[0])])[0]
            # translated_images_b = self.decoders[1]([random_style_codes[1], tf.gather(encoded_contents, random_source_domain[1])])[0]
            # translated_images_c = self.decoders[2]([random_style_codes[2], tf.gather(encoded_contents, random_source_domain[2])])[0]
            # translated_images_d = self.decoders[3]([random_style_codes[3], tf.gather(encoded_contents, random_source_domain[3])])[0]
            # translated_images = tf.stack([translated_images_a, translated_images_b, translated_images_c, translated_images_d], axis=0)

            # 4. encode again the cross-domain encoded
            encoded_translated_contents = [self.content_encoders[i](translated_images[i])
                                           for i in range(number_of_domains)]
            encoded_translated_contents = tf.stack(encoded_translated_contents)
            # print("encoded_translated_contents", encoded_translated_contents)
            # print("random_source_domain", random_source_domain)
            encoded_translated_contents = tf.gather(encoded_translated_contents, random_source_domain)
            encoded_translated_styles = [self.style_encoders[i](translated_images[i])
                                         for i in range(number_of_domains)]

            # 5. decode once more (this is not used in the tf implementation of munit)
            # decoded_translated_images = [self.decoders[i]([tf.gather(encoded_styles, i),
            #                                                tf.gather(encoded_translated_contents,
            #                                                          tf.gather(random_source_domain, i))])[0]
            #                              for i in range(number_of_domains)]

            # 6. discriminate the input images (real, then fake from cross-domain translation)
            predicted_patches_real = [self.discriminators[i](batch[i]) for i in range(number_of_domains)]
            predicted_patches_fake = [self.discriminators[i](translated_images[i]) for i in range(number_of_domains)]

            d_loss = self.discriminator_loss(predicted_patches_real, predicted_patches_fake)
            g_loss = self.generator_loss(predicted_patches_fake, decoded_images, batch, encoded_translated_styles,
                                         random_style_codes, encoded_translated_contents, encoded_contents)

        # 7. apply the gradients to the models
        discriminator_gradients = [tape.gradient(d_loss["total"][i], self.discriminators[i].trainable_variables)
                                   for i in range(number_of_domains)]
        generator_gradients = [tape.gradient(g_loss["total"][i], self.generators[i].trainable_variables)
                               for i in range(number_of_domains)]
        for i in range(number_of_domains):
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients[i],
                                                             self.discriminators[i].trainable_variables))
            self.generator_optimizer.apply_gradients(zip(generator_gradients[i],
                                                         self.generators[i].trainable_variables))

        # write statistics of the training step
        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                fake_loss = real_loss = total_loss = 0
                for i in range(number_of_domains):
                    fake_loss += d_loss["fake"][i]
                    real_loss += d_loss["real"][i]
                    total_loss += d_loss["total"][i]
                tf.summary.scalar("fake_loss", tf.reduce_mean(fake_loss / number_of_domains), step=step // update_steps)
                tf.summary.scalar("real_loss", tf.reduce_mean(real_loss / number_of_domains), step=step // update_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss / number_of_domains),
                                  step=step // update_steps)

        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                adversarial_loss = same_domain_loss = style_loss = content_loss = cross_domain_loss = total_loss = 0
                for i in range(number_of_domains):
                    adversarial_loss += g_loss["adversarial"][i]
                    same_domain_loss += g_loss["same-domain"][i]
                    style_loss += g_loss["style"][i]
                    content_loss += g_loss["content"][i]
                    # cross_domain_loss += g_loss["cross-domain"][i]
                    total_loss += g_loss["total"][i]
                tf.summary.scalar("adversarial_loss", tf.reduce_mean(adversarial_loss), step=step // update_steps)
                tf.summary.scalar("same_domain_loss", tf.reduce_mean(same_domain_loss), step=step // update_steps)
                tf.summary.scalar("style_loss", tf.reduce_mean(style_loss), step=step // update_steps)
                tf.summary.scalar("content_loss", tf.reduce_mean(content_loss), step=step // update_steps)
                # tf.summary.scalar("cross_domain_loss", tf.reduce_mean(cross_domain_loss),
                #                   step=step // update_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss), step=step // update_steps)

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
            if c == number_of_examples - 1:
                source_index = ensure_inside_range(2)
                target_index = source_index

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
        title = ["Input", "Reconstructed", "Random Style", "Translated", "Target"]
        num_rows = len(examples)
        num_cols = len(title)

        if step is not None:
            if step == 1:
                step = 0
            title[1] += f" (step {step / 1000}k)"
            title[2] += f" (step {step / 1000}k)"
            title[3] += f" (step {step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i in range(num_rows):
            source_image = examples[i][0]
            target_image = examples[i][1]
            random_target_style = examples[i][2]
            source_domain = examples[i][3]
            target_domain = examples[i][4]
            source_image_content = self.content_encoders[source_domain](source_image[tf.newaxis, ...])
            source_image_style = self.style_encoders[source_domain](source_image[tf.newaxis, ...])
            for j in range(num_cols):
                index = i * num_cols + j + 1
                plt.subplot(num_rows, num_cols, index)
                plt.title(title[j])
                image = None
                if j == 0:
                    image = source_image
                elif j == num_cols - 1:
                    image = target_image
                elif j == 1:
                    image = self.decoders[source_domain]([source_image_style, source_image_content])[0]
                elif j == 2:
                    image = self.decoders[source_domain]([random_target_style, source_image_content])[0]
                elif j == 3:
                    image = self.decoders[target_domain]([random_target_style, source_image_content])[0]
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

                for i in range(batch_end - batch_start):
                    decoder = self.decoders[target_domains_slice[i]]
                    style_encoder = self.style_encoders[target_domains_slice[i]]
                    content_encoder = self.content_encoders[source_domains_slice[i]]
                    fake_images_slice = decoder([style_encoder(tf.expand_dims(source_images_slice[i], 0)),
                                                 content_encoder(tf.expand_dims(source_images_slice[i], 0))])[0]
                    fake_images[batch_start + i] = fake_images_slice

            logging.debug(
                f"Generated all {number_of_examples} fake images for {dataset_name} dataset, which occupy {fake_images.nbytes / 1024 / 1024} MB.")
            return target_images, tf.constant(fake_images)

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def generate_images_from_dataset(self, dataset, step, num_images=None):
        pass

    def debug_discriminator_output(self, batch, image_path):
        pass
