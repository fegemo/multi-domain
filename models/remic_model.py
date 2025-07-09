import os
from abc import ABC, abstractmethod
from functools import reduce

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility import dataset_utils, io_utils
from .munit_model import MunitModel
from .networks import remic_generator, remic_discriminator, remic_style_encoder, remic_unified_content_encoder


class RemicModel(MunitModel):
    """
    Implements ReMIC model, which is a modification of MUNIT. The main difference is that the content encoder is shared.
    There is no reference implementation. I sent an email to the authors of the paper and they said they could not
    share it.
    """

    def __init__(self, config, export_additional_training_endpoint=False):
        self.unified_content_encoder = None
        self.style_encoders = None
        self.decoders = None
        self.generators = None
        self.discriminators = None
        super().__init__(config, export_additional_training_endpoint)
        self.lambda_image_consistency = config.lambda_l1
        self.lambda_latent_consistency = config.lambda_latent_reconstruction
        self.lambda_image_reconstruction = config.lambda_cyclic_reconstruction
        if config.input_dropout == "none":
            self.sampler = NoDropoutSampler(config)
        elif config.input_dropout == "original":
            self.sampler = UniformDropoutSampler(config)
        elif config.input_dropout == "conservative":
            self.sampler = ConservativeDropoutSampler(config)
        elif config.input_dropout == "curriculum":
            self.sampler = CurriculumDropoutSampler(config)
        else:
            raise ValueError(f"The provided {config.input_dropout} type for input dropout has not been implemented.")
        self.l1_loss = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred))

    def create_inference_networks(self):
        config = self.config
        image_size = config.image_size
        inner_channels = config.inner_channels
        number_of_domains = config.number_of_domains
        domain_letters = [name[0].upper() for name in config.domains]
        if config.generator in ["remic", ""]:
            unified_content_encoder = remic_unified_content_encoder("Unified", image_size, inner_channels,
                                                                    number_of_domains)
            style_encoders = [remic_style_encoder(s, image_size, inner_channels) for s in domain_letters]
            decoders = [remic_generator(s, inner_channels) for s in domain_letters]

            single_image_input = tf.keras.layers.Input(shape=(image_size, image_size, inner_channels))
            all_images_input = tf.keras.layers.Input(shape=(number_of_domains, image_size, image_size, inner_channels))
            x = single_image_input
            x_all = all_images_input
            generators = [tf.keras.Model(inputs=(x, x_all),
                                         outputs=decoders[d]([style_encoders[d](x), unified_content_encoder(x_all)]),
                                         name=f"Generator{domain_letters[d]}")
                          for d in range(number_of_domains)]
            # generators is a list of "virtual" models, as in MUNIT (see MunitModel implementation)

            self.unified_content_encoder = unified_content_encoder
            self.style_encoders = style_encoders
            self.decoders = decoders
            self.generators = generators

            return {
                "unified_content_encoder": unified_content_encoder,
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
        if config.discriminator in ["remic", ""]:
            discriminators = [remic_discriminator(s, image_size, inner_channels, scales)
                              for s in domain_letters]
            self.discriminators = discriminators
            return {
                "discriminators": discriminators
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type of discriminator has not been implemented")

    def generator_loss(self, predicted_patches_fake,
                       decoded_images, source_images, input_keep_mask,
                       original_contents, recoded_contents, random_styles, recoded_styles,
                       decoded_images_with_random_style):
        # predicted_patches_fake (shape=[d, ds, b, ?, ?, 1])
        number_of_domains = self.config.number_of_domains
        discriminator_scales = self.config.discriminator_scales

        # loss from the discriminator
        adversarial_loss = [
            [self.lsgan_loss(tf.ones_like(predicted_patches_fake[d][ds]), predicted_patches_fake[d][ds])
             for ds in range(discriminator_scales)]
            for d in range(number_of_domains)]
        adversarial_loss = tf.reduce_mean(adversarial_loss, axis=1)
        # adversarial_loss (shape=[d])

        # image consistency loss (trying to reconstruct the input images from their original content and style)
        decoded_images = tf.transpose(decoded_images, [1, 0, 2, 3, 4])
        visible_decoded_images = decoded_images * input_keep_mask
        visible_source_images = source_images * input_keep_mask
        image_consistency_loss = [self.l1_loss(visible_source_images[:, d], visible_decoded_images[:, d])
                                  for d in range(number_of_domains)]

        # latent consistency loss (trying to reconstruct the real content and random style codes)
        content_consistency_loss = self.l1_loss(original_contents, recoded_contents)
        style_consistency_loss = [self.l1_loss(random_styles[d], recoded_styles[d])
                                  for d in range(number_of_domains)]

        # image reconstruction loss (trying to reconstruct the original images from their original content
        # and random style)
        source_images = tf.transpose(source_images, [1, 0, 2, 3, 4])
        image_reconstruction_loss = [self.l1_loss(source_images[d], decoded_images_with_random_style[d])
                                     for d in range(number_of_domains)]

        # l2 regularization loss
        l2_regularization = [tf.reduce_sum(self.decoders[i].losses) for i in range(number_of_domains)]

        total_loss = [adversarial_loss[d] +
                      self.lambda_image_consistency * image_consistency_loss[d] +
                      self.lambda_latent_consistency * content_consistency_loss +
                      self.lambda_latent_consistency * style_consistency_loss[d] +
                      self.lambda_image_reconstruction * image_reconstruction_loss[d] +
                      l2_regularization[d]
                      for d in range(number_of_domains)]
        return {
            "adversarial": adversarial_loss, "image-consistency": image_consistency_loss,
            "style-consistency": style_consistency_loss, "content-consistency": content_consistency_loss,
            "image-reconstruction": image_reconstruction_loss,
            "l2-regularization": l2_regularization,
            "total": total_loss
        }

    def discriminator_loss(self, predicted_patches_real, predicted_patches_fake):
        return super().discriminator_loss(predicted_patches_real, predicted_patches_fake)

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        """
        These steps were inferred from the ReMIC paper (there is no reference implementation):
        0. generates random style codes
        1. encodes content from the inputs
        2. encodes style from the inputs
        3. decodes with original content/style codes
           - image consistency loss
        4. decodes with original contents, random styles
        5. encodes content of the generated images in (4)
        6. encodes style of the generated images in (4)
           - latent consistency loss
        7. discriminates real and generated images in (4)
           - adversarial loss
        8. calculates reconstruction loss (4)
           - reconstruction loss
        :param batch:
        :param step:
        :param evaluate_steps:
        :param t:
        :return:
        """
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = self.config.number_of_domains, batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # a. determines random domains to be dropped out
        source_images, input_keep_mask = self.sampler.sample(batch, t)
        # source_images (shape=[b, d, s, s, c])
        # input_dropout_mask (shape=[b, d])
        input_keep_mask = input_keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # input_keep_mask (shape=[b, d] -> [b, d, 1, 1, 1])

        visible_source_images = source_images * input_keep_mask
        # visible_source_images (shape=[b, d, s, s, c])

        # 0. generates random style codes
        random_style_codes = [tf.random.normal([batch_size, 8]) for _ in range(number_of_domains)]
        # random_style_codes (shape=[d, b, 8])

        with tf.GradientTape(persistent=True) as tape:
            # 1. encode the input images to get their content and style of the visible images
            encoded_contents = self.unified_content_encoder(visible_source_images)
            encoded_styles = [self.style_encoders[d](visible_source_images[:, d]) for d in range(number_of_domains)]
            # encoded_contents (shape=[b, 16, 16, 256])
            # encoded_styles (shape=[d, b, 8])

            # 2. decode de encoded input images (using original content and style) to perfectly reconstruct them
            # this is for ReMIC's "image consistency loss"
            decoded_images = [self.decoders[d]([encoded_styles[d], encoded_contents])[0]
                              for d in range(number_of_domains)]
            # decoded_images (shape=[d, b, s, s, c])

            # 3. decode the encoded input images (with a random style) to generate fake images
            # this is for ReMIC's "latent consistency loss", "adversarial loss" and "reconstruction loss"
            decoded_images_with_random_style = [self.decoders[d]([random_style_codes[d], encoded_contents])[0]
                                                for d in range(number_of_domains)]
            # decoded_images_with_random_style (shape=[d, b, s, s, c])

            # 4. encode the images generated with random style
            # this is for ReMIC's "latent consistency loss"
            encoded_contents_with_random_style = self.unified_content_encoder(
                tf.transpose(decoded_images_with_random_style, [1, 0, 2, 3, 4]))
            # encoded_contents_with_random_style (shape=[b, 16, 16, 256])
            encoded_style_with_random_style = [self.style_encoders[d](decoded_images_with_random_style[d])
                                               for d in range(number_of_domains)]

            # 5. discriminates the images generated with random style
            # this is for ReMIC's "adversarial loss"
            predicted_patches_real = [self.discriminators[i](visible_source_images[:, i])
                                      for i in range(number_of_domains)]
            predicted_patches_fake = [self.discriminators[i](decoded_images_with_random_style[i])
                                      for i in range(number_of_domains)]
            # predicted_patches_xxxx (shape=[d, b, ds, ?, ?, 1]) where ds are the discriminator scales and
            # ?, ? are the dimensions of the patches (different for each scale)

            # calculates the loss functions
            d_loss = self.discriminator_loss(predicted_patches_real, predicted_patches_fake)
            g_loss = self.generator_loss(
                # for adversarial loss
                predicted_patches_fake,
                # for image consistency loss
                decoded_images, source_images, input_keep_mask,
                # for latent consistency loss
                encoded_contents, encoded_contents_with_random_style,
                random_style_codes, encoded_style_with_random_style,
                # for image reconstruction loss
                decoded_images_with_random_style)

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
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, all_discriminator_trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, all_generator_trainable_variables))

        # writes statistics of the training step
        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                fake_loss = real_loss = weight_loss = total_loss = 0
                for i in range(number_of_domains):
                    fake_loss += d_loss["fake"][i]
                    real_loss += d_loss["real"][i]
                    weight_loss += d_loss["l2-regularization"][i]
                    total_loss += d_loss["total"][i]
                tf.summary.scalar("fake_loss", tf.reduce_mean(fake_loss / number_of_domains), step=step // evaluate_steps)
                tf.summary.scalar("real_loss", tf.reduce_mean(real_loss / number_of_domains), step=step // evaluate_steps)
                tf.summary.scalar("weight_loss", tf.reduce_mean(weight_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss / number_of_domains),
                                  step=step // evaluate_steps)

        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                adversarial_loss = image_consistency_loss = style_loss = 0
                image_reconstruction_loss = weight_loss = total_loss = 0
                content_loss = g_loss["content-consistency"]
                for i in range(number_of_domains):
                    adversarial_loss += g_loss["adversarial"][i]
                    image_consistency_loss += g_loss["image-consistency"][i]
                    style_loss += g_loss["style-consistency"][i]
                    image_reconstruction_loss += g_loss["image-reconstruction"][i]
                    weight_loss += g_loss["l2-regularization"][i]
                    total_loss += g_loss["total"][i]
                tf.summary.scalar("adversarial_loss", tf.reduce_mean(adversarial_loss), step=step // evaluate_steps)
                tf.summary.scalar("image_consistency_loss", tf.reduce_mean(image_consistency_loss),
                                  step=step // evaluate_steps)
                tf.summary.scalar("style_loss", tf.reduce_mean(style_loss), step=step // evaluate_steps)
                tf.summary.scalar("content_loss", tf.reduce_mean(content_loss), step=step // evaluate_steps)
                tf.summary.scalar("image_reconstruction_loss", tf.reduce_mean(image_reconstruction_loss),
                                  step=step // evaluate_steps)
                tf.summary.scalar("weight_loss", tf.reduce_mean(weight_loss), step=step // evaluate_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss), step=step // evaluate_steps)

    def select_examples_for_visualization(self, train_ds, test_ds):
        number_of_domains = self.config.number_of_domains
        number_of_examples = 3
        train_examples = []
        test_examples = []

        train_ds_iter = train_ds.unbatch().take(number_of_examples).as_numpy_iterator()
        test_ds_iter = test_ds.shuffle(self.config.test_size).unbatch().take(number_of_examples).as_numpy_iterator()
        if number_of_domains == 4:
            # if there are 4 domains, we drop 1, 2 or 3 domains, always choosing the permutation that drops the
            # later domains (e.g., for 1, drop right; for 2, drop right/front; for 3, drop right/front/left)

            keep_masks = [
                [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]
            ]
            for c in range(number_of_examples):
                keep_mask = keep_masks[c]
                keep_mask = tf.cast(keep_mask, dtype="float32")

                train_batch = next(train_ds_iter)
                train_example = (train_batch, keep_mask)

                test_batch = next(test_ds_iter)
                test_example = (test_batch, keep_mask)

                train_examples.append(train_example)
                test_examples.append(test_example)
        else:
            # if domains != 4, we randomly select the permutation of domains to drop, by starting with a single
            # domain and increasing by one at each row
            null_list = dataset_utils.create_domain_permutation_list(number_of_domains)
            for c in range(number_of_examples):
                to_drop = (c % (number_of_domains - 1)) + 1
                domain_permutation = null_list[to_drop - 1][0]
                # domain_permutation (shape=[d], int32)
                keep_mask = tf.cast(domain_permutation, dtype="float32")

                train_batch = next(train_ds_iter)
                train_example = (train_batch, keep_mask)

                test_batch = next(test_ds_iter)
                test_example = (test_batch, keep_mask)

                train_examples.append(train_example)
                test_examples.append(test_example)

        return train_examples + test_examples

    def preview_generated_images_during_training(self, examples, save_name, step):
        number_of_domains = self.config.number_of_domains
        image_size = self.config.image_size
        channels = self.config.inner_channels
        domains = self.config.domains_capitalized
        titles = domains + [f"Gener. {d}" for d in domains]
        num_rows = len(examples)
        num_cols = len(titles)

        if step is not None:
            if step == 1:
                step = 0
            for d in range(1, number_of_domains + 1):
                titles[-1 * d] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i, example in enumerate(examples):
            source_images, keep_mask = example
            # source_images (d, s, s, c)
            # keep_mask (d)

            keep_mask = keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
            # keep_mask (d, 1, 1, 1)

            visible_source_images = source_images * keep_mask
            # visible_source_images (d, s, s, c)

            encoded_contents = self.unified_content_encoder(visible_source_images[tf.newaxis, ...])
            encoded_styles = [self.style_encoders[d](visible_source_images[d][tf.newaxis, ...])
                              for d in range(number_of_domains)]
            # encoded_contents (1, 16, 16, 256)
            # encoded_styles (d, 1, 8)

            decoded_images = [self.decoders[d]([encoded_styles[d], encoded_contents])[0]
                              for d in range(number_of_domains)]

            contents = [*tf.squeeze(visible_source_images), *tf.squeeze(decoded_images)]

            for j in range(num_cols):
                idx = i * num_cols + j + 1
                plt.subplot(num_rows, num_cols, idx)
                if i == 0:
                    plt.title(titles[j], fontdict={"fontsize": 24})
                plt.imshow(contents[j] * 0.5 + 0.5)
                plt.axis("off")

        figure.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        return figure

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        uniform_sampler = UniformDropoutSampler(self.config)

        def initialize_random_examples_from_dataset(dataset):
            batch = list(dataset.unbatch().take(num_images))
            domain_images = tf.transpose(batch, [1, 0, 2, 3, 4])
            domain_images, keep_list = uniform_sampler.sample(domain_images, 0)

            # gets the index of the first zero in the keep_list for each element in the batch
            possible_target_domain = tf.argmin(keep_list, axis=1, output_type="int32")

            return domain_images, keep_list[..., tf.newaxis, tf.newaxis, tf.newaxis], possible_target_domain

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        batch_size = self.config.batch

        def generate_images_from_example_indices(example_indices):
            domain_images, keep_mask, possible_target_domain = example_indices
            # domain_images (b, d, s, s, c)
            # keep_mask (b, d, 1, 1, 1)

            visible_source_images = domain_images * keep_mask
            # visible_source_images (b, d, s, s, c)

            encoded_contents = self.unified_content_encoder.predict(visible_source_images, batch_size=batch_size,
                                                                    verbose=0)
            encoded_styles = [self.style_encoders[d].predict(visible_source_images[:, d], batch_size=batch_size,
                                                             verbose=0)
                              for d in range(self.config.number_of_domains)]
            # encoded_contents (b, 16, 16, 256)
            # encoded_styles (d, b, 8)

            decoded_images = [self.decoders[d].predict([encoded_styles[d], encoded_contents], batch_size=batch_size,
                                                       verbose=0)[0]
                              for d in range(self.config.number_of_domains)]

            fake_images = tf.gather(tf.transpose(decoded_images, [1, 0, 2, 3, 4]), possible_target_domain, axis=1,
                                    batch_dims=1)
            real_images = tf.gather(domain_images, possible_target_domain, batch_dims=1)
            return real_images, fake_images

        return {
            "train": generate_images_from_example_indices(example_indices_for_evaluation["train"]),
            "test": generate_images_from_example_indices(example_indices_for_evaluation["test"])
        }

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        null_list = dataset_utils.create_input_dropout_index_list(list(range(1, number_of_domains)), number_of_domains)
        num_cols = number_of_domains * 2  # generate images with the original style, then with a random style code
        num_rows = reduce(lambda acc, v: acc + len(v), null_list[0], 0)

        permutations_for_each_target_domain = [reduce(lambda acc, p: acc + p, null_list[d], [])
                                          for d in range(number_of_domains)]
        oh_to_readable_domains = lambda oh: "+".join([self.config.domains[l][0] for l in range(len(oh)) if oh[l] != 0])
        # permutations_for_each_target_domain (d, x, d), with x being the number of permutations for the target domain,
        # which is the number of rows of the image
        for c, domain_images in tqdm(enumerated_dataset, total=num_images):
            # each character will appear in a separate png file
            image_path = os.sep.join([base_image_path, f"{c:04d}_at_step_{step}.png"])
            fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))

            generated_with_original_style = []
            generated_with_random_style = []
            for i, _ in enumerate(permutations_for_each_target_domain[0]):
                # each row will have a different permutation of domains to drop. For instance,
                # for d=4, row 0 will feature [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]
                #          row 1 will feature [0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]
                #          row 2 will feature [0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]
                # ....
                generated_with_original_style.append([])
                generated_with_random_style.append([])
                for j in range(number_of_domains):
                    domain_permutation = 1. - tf.constant(permutations_for_each_target_domain[j][i], dtype="float32")
                    keep_mask = tf.cast(domain_permutation, dtype="float32")
                    visible_domain_images = domain_images * keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
                    content_code = self.unified_content_encoder(visible_domain_images[tf.newaxis, ...])

                    # generates the image using the original style code
                    style_code = self.style_encoders[j](domain_images[j][tf.newaxis, ...])
                    decoded_image = self.decoders[j]([style_code, content_code])[0]
                    generated_with_original_style[-1].append(tf.squeeze(decoded_image))

                    # draws the image with the original style
                    domain_name = self.config.domains_capitalized[j]
                    from_domains_abbr = oh_to_readable_domains(domain_permutation)
                    title = f"{domain_name} ({from_domains_abbr}, ori)"

                    plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
                    plt.title(title, fontdict={"fontsize": 24})
                    plt.imshow(generated_with_original_style[i][j] * 0.5 + 0.5)
                    plt.axis("off")

                    # generates the image using a random style code
                    random_style_code = tf.random.normal([1, 8])
                    decoded_image = self.decoders[j]([random_style_code, content_code])[0]
                    generated_with_random_style[-1].append(tf.squeeze(decoded_image))

                    title = f"{domain_name} ({from_domains_abbr}, rnd)"
                    plt.subplot(num_rows, num_cols, i * num_cols + j + number_of_domains + 1)
                    plt.title(title, fontdict={"fontsize": 24})
                    plt.imshow(generated_with_random_style[i][j] * 0.5 + 0.5)
                    plt.axis("off")

            fig.tight_layout()
            plt.savefig(image_path, transparent=True)
            plt.close(fig)

    def debug_discriminator_output(self, batch, image_path):
        discriminator_scales = self.config.discriminator_scales
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch (b, d, s, s, c)
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = batch_shape[1], batch_shape[0], batch_shape[2], \
            batch_shape[4]

        ensure_inside_range = lambda x: tf.math.floormod(x, number_of_domains)
        target_domains = [ensure_inside_range(x) for x in range(batch_size)]
        real_images = tf.gather(batch, target_domains, axis=1, batch_dims=1)
        fake_images = []
        for i in range(batch_size):
            keep_mask = tf.one_hot(target_domains[i], number_of_domains, on_value=0.0, off_value=1.0)
            keep_mask = keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
            visible_source_images = batch[i] * keep_mask
            # visible_source_images (d, s, s, c)

            content_code = self.unified_content_encoder(visible_source_images[tf.newaxis, ...])
            random_style_code = tf.random.normal([1, 8])
            fake_image = self.decoders[target_domains[i]]([
                random_style_code,
                content_code
            ])[0]
            fake_images.append(fake_image)
        fake_images = tf.concat(fake_images, axis=0)

        # gets the result of discriminating the real and fake (translated) images
        real_patches = [self.discriminators[target_domains[i]](real_images[i][tf.newaxis, ...])
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
        titles = ["Real", *discriminator_titles, "Imputed", *discriminator_titles]
        num_cols = len(titles)
        num_rows = batch_size.numpy()

        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows, num_cols, (i * num_cols) + j + 1)
                subplot_title = ""
                if i == 0:
                    subplot_title = titles[j]
                plt.title(subplot_title, fontdict={"fontsize": 24})

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


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch, t):
        """
        Returns the batch with shape (b, d, s, s, c) and a keep mask with shape (b, d) -> float32
        indicating, for each sample in the batch, which domains are kept (1) and which are dropped (0).
        :param batch: batch of samples with shape (d, b, s, s, c)
        :param t: the progress of the training between [0, 1]
        :return: batch, keep_mask
        """
        pass


class UniformDropoutSampler(ExampleSampler):
    """
    Input Dropout Sampler that implements the input dropout strategy that seems to be presented in the ReMIC paper
    (but it is not clear). It is based on the CollaGAN implementation.
    This selects a uniform random number of domains to drop, then selects a random domain combination from that pool.
    """

    def __init__(self, config):
        super().__init__(config)
        # a list shape=(to_drop, ?, d) that is,
        #     for each possible number of dropped inputs (first dimension): all permutations of an int array that
        #     (a) nullifies a number of inputs equal to 1, 2 or 3 (in case of 4 domains).
        self.null_list = tf.ragged.constant(dataset_utils.create_domain_permutation_list(self.config.number_of_domains),
                                            ragged_rank=1, dtype="float32")

    def sample(self, batch, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        # reorders the batch from [d, b, s, s, c] to [B, D, s, s, c]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])

        # finds a random number of domains to drop
        # (shape=[b]), eg, [1, 2, 1, 3] (in the range [1, number_of_domains[)
        number_of_domains_to_drop = self.select_number_of_inputs_to_drop(batch_size, t)

        # repeats the null_list for each example in the batch, so we can tf.gather it later
        repeated_null_list = tf.tile(self.null_list[tf.newaxis, ...], [batch_size, 1, 1, 1])
        # repeated_null_list (shape=[b, to_drop, ?, d])

        # selects the input dropout mask for each example in the batch
        input_dropout_mask_options = tf.gather(repeated_null_list, number_of_domains_to_drop - 1, batch_dims=1)

        # selects a random permutation for each example in the batch
        # (b/c of a tf bug, we need to generate float32 values than truncate them to int32)
        # (more: the bug is that tf.random.uniform does not support int values with 1-D maxval)
        random_permutation_index = tf.random.uniform([batch_size],
                                                     maxval=tf.cast(input_dropout_mask_options.row_lengths(),
                                                                    "float32"))
        random_permutation_index = tf.cast(random_permutation_index, dtype="int32")

        # input_dropout_mask (shape=[b, d])
        input_dropout_mask = tf.gather(input_dropout_mask_options, random_permutation_index, batch_dims=1)
        # input_keep_mask (shape=[b, d])
        input_keep_mask = 1. - input_dropout_mask

        # returns the batch and an input_keep_mask (inverse of the dropout mask)
        # the mask is a 0/1 tensor
        return batch, input_keep_mask

    def select_number_of_inputs_to_drop(self, batch_size, t):
        number_of_domains = self.config.number_of_domains
        return tf.random.uniform(shape=[batch_size], minval=1, maxval=number_of_domains, dtype="int32")


class ConservativeDropoutSampler(UniformDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, t):
        # number_of_domains = self.config.number_of_domains
        u = tf.random.uniform(shape=[batch_size])
        # returns a random number of domains to drop with higher chances of dropping fewer domains
        # 10% of the time, drop 3 inputs
        # 30% of the time, drop 2 inputs
        # 60% of the time, drop 1 inputs
        return tf.where(u < 0.1, 3, tf.where(u < 0.4, 2, 1))


class CurriculumDropoutSampler(UniformDropoutSampler):
    def __init__(self, config):
        super().__init__(config)

    def select_number_of_inputs_to_drop(self, batch_size, t):
        # start with easy (missing 1) samples, then move to harder ones
        # until 17% of the training, drop 1 inputs
        # until 33% of the training, drop 2 inputs
        # until 50% of the training, drop 3 inputs
        # remainder 50% of the training, drop randomly
        n = super().select_number_of_inputs_to_drop(batch_size, t)
        return tf.where(t < 0.166667, 1, tf.where(t < 0.333333, 2, tf.where(t < 0.5, 3, n)))


class NoDropoutSampler(ExampleSampler):
    def sample(self, batch, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch (b, d, s, s, c)

        # selects a random single missing domain for each sample in the batch
        missing_domains = tf.random.uniform([batch_size], minval=0, maxval=number_of_domains, dtype="int32")
        input_keep_mask = tf.one_hot(missing_domains, number_of_domains, on_value=0., off_value=1.)

        return batch, input_keep_mask
