import os
from functools import reduce

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility import keras_utils, palette_utils, io_utils, dataset_utils
from utility.functional_utils import listify
from utility.keras_utils import LinearAnnealingScheduler, NoopAnnealingScheduler, create_random_inpaint_mask, \
    NoopInpaintMaskGenerator, ConstantInpaintMaskGenerator, RandomInpaintMaskGenerator, CurriculumInpaintMaskGenerator
from .networks import resblock, munit_discriminator_multi_scale
from .remic_model import RemicModel


class SpriteEditorModel(RemicModel):
    def __init__(self, config):
        self.generator = None
        self.discriminators = None
        super().__init__(config, export_additional_training_endpoint=True)
        self.lambda_kl = config.lambda_kl
        self.lambda_adversarial = config.lambda_adversarial
        self.lambda_reconstruction = config.lambda_reconstruction
        self.lambda_palette = config.lambda_palette

        self.generator = self.inference_networks["generator"]
        self.discriminators = self.training_only_networks["discriminators"]
        self.diversity_encoder = self.inference_networks["diversity-encoder"]

        # the third param is the target palette, so we skip it if we're not using palette quantization
        self.gen_supplier = keras_utils.SkipParamsSupplier([2] if not config.palette_quantization else None)

        if config.palette_quantization and config.annealing != "none":
            self.annealing_scheduler = LinearAnnealingScheduler(config.temperature, [self.generator.quantization])
        else:
            self.annealing_scheduler = NoopAnnealingScheduler()

        if config.inpaint_mask == "none":
            self.mask_creator = NoopInpaintMaskGenerator()
        elif config.inpaint_mask == "random":
            self.mask_creator = RandomInpaintMaskGenerator()
        elif config.inpaint_mask == "constant":
            self.mask_creator = ConstantInpaintMaskGenerator()
        elif config.inpaint_mask == "curriculum":
            self.mask_creator = CurriculumInpaintMaskGenerator()

        self.diversity_encoder_optimizer = None

    def fit(self, train_ds, test_ds, steps, update_steps, callbacks=[], starting_step=0):
        self.diversity_encoder_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.lr, beta_1=0.5, beta_2=0.999)
        super().fit(train_ds, test_ds, steps, update_steps, callbacks=callbacks, starting_step=starting_step)

    def create_inference_networks(self):
        config = self.config
        if config.generator in ["monolith", ""]:
            generator = build_monolith_generator(config)
            diversity_encoder = build_diversity_encoder(config)
            return {
                "generator": generator,
                "diversity-encoder": diversity_encoder
            }
        else:
            raise ValueError(f"The provided {config.generator} type of generator has not been implemented")

    def create_training_only_networks(self):
        config = self.config
        image_size = config.image_size
        inner_channels = config.inner_channels
        scales = config.discriminator_scales
        domain_letters = [name[0].upper() for name in config.domains]
        if config.discriminator in ["munit", ""]:
            discriminators = [build_munitlike_discriminator(s, image_size, inner_channels, scales)
                              for s in domain_letters]
            self.discriminators = discriminators
            return {
                "discriminators": discriminators
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type of discriminator has not been implemented")

    def encode(self, images, domain_availability, training=True):
        """
        Encodes the input images into latent codes using the diversity encoder.
        :param images: the input images to be encoded. The shape is [b, d, s, s, c] where b is the batch size,
            d is the number of domains, s is the size of the images, and c is the number of channels.
        :param domain_availability: a multi-hot encoded vector indicating which domains are available.
            The shape is [b, d].
        :param training: whether the model is in training mode or not.
        :return: the extracted latent codes from the diversity encoder, their mean and log-variance.
        """
        mu, logvar = self.diversity_encoder([images, domain_availability], training=training)
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(std.shape)
        z = mu + eps * std
        # z (shape=[b, noise_length])
        return z, mu, logvar

    def generator_loss(self, fake_predicted_patches, generated_images, target_images, input_dropout_mask,
                       ec_mean, ec_log_var, random_codes, recovered_codes_mean, target_palettes, temperature,
                       batch_shape):
        """
        Calculates the generator loss for the Sprite Editor model.
        :param fake_predicted_patches: discriminator outputs for the generated images. The shape depends on the number
            of domains and discriminator scales: [d, ds] x [b, ?, ?, 1] where d is the number of domains,
            ds is the number of discriminator scales, b is the batch size, and ? is the size of discriminated patches.
        :param generated_images: the fake images generated by the generator. The shape depends on the number of
            generator scales: [gs] x [b, d, ?, ?, c] where gs is the generator scales, b is the batch size, d is
            the number of domains, ? is the size of the generated images, and c is the number of channels.
        :param target_images: the ground truth images for the generator. The shape depends on the number of
            generator scales: [gs] x [hb, d, ?, ?, c].
        :param input_dropout_mask:
        :param ec_mean: the mean of the extracted codes from the source images. The shape is [hb, noise_length]
        :param ec_log_var: the log-variance of the extracted codes from the source images. The shape is [hb, noise_length]
        :param random_codes: the random codes used to generate the images. The shape is [hb, noise_length]
        :param recovered_codes_mean: the codes recovered from the images generated with random codes. The shape is
            [hb, noise_length]
        :param target_palettes: the target palette for the generated images. The shape is [b, (n), c] where
        b is the batch size, (n) is the number of colors in the palette, and c is the number of channels.
        :param temperature: the temperature for the palette quantization.
        :param batch_shape: the original shape of the batch as it comes inside the train_step method.
        :return: a dictionary with keys for each loss term and the total loss.
        """
        generator_scales = self.config.generator_scales
        discriminator_scales = self.config.discriminator_scales
        number_of_domains = self.config.number_of_domains
        batch_size, image_size, channels = batch_shape[1], batch_shape[2], batch_shape[4]
        half_batch_size = batch_size // 2

        # 1. calculates the adversarial loss (lsgan)
        adversarial_loss = [
            [self.lsgan_loss(tf.ones_like(fake_predicted_patches[d][ds]), fake_predicted_patches[d][ds])
             for ds in range(discriminator_scales)]
            for d in range(number_of_domains)]
        # adversarial_loss (shape=[d, ds])
        adversarial_loss = tf.reduce_mean(adversarial_loss)
        # adversarial_loss (shape=[])

        # 2. calculates reconstruction (l1) loss between the target and the generated images
        reconstruction_loss = [tf.reduce_mean(tf.abs(target_images[gs] - generated_images[gs][:half_batch_size]))
                               for gs in range(generator_scales)]
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        # 3. calculates the latent reconstruction loss
        latent_reconstruction_loss = tf.reduce_mean(tf.abs(random_codes - recovered_codes_mean))

        # 4. calculates the palette coverage loss
        palette_coverage_loss = palette_utils.calculate_palette_coverage_loss(generated_images[0], target_palettes,
                                                                              temperature)

        # 5. kullback leibler divergence loss for the extracted codes
        kl_loss = -0.5 * tf.reduce_mean(1 + ec_log_var - tf.square(ec_mean) - tf.exp(ec_log_var))

        # 6. calculate the total weighted loss
        total_loss = self.lambda_adversarial * adversarial_loss + \
                     self.lambda_reconstruction * reconstruction_loss + \
                     self.lambda_latent_reconstruction * latent_reconstruction_loss + \
                     self.lambda_kl * kl_loss + \
                     self.lambda_palette * palette_coverage_loss

        # returns the weighted total generator loss and the individual terms
        return {"adversarial": adversarial_loss, "reconstruction": reconstruction_loss,
                "latent-reconstruction": latent_reconstruction_loss, "palette-coverage": palette_coverage_loss,
                "kl": kl_loss,
                "total": total_loss}

    # @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        """
        Performs a single training step for the Sprite Editor model.

        encoder:
          1. encodes the source images into extracted_code
          2. loss: Kullback-Leibler for the extracted code and N(0,1)
          3. encodes the images generated using the random code into recovered_code
          4. loss: l1 between the random code N(0,1) with the recovered_code

        generator:
          1. generates from source + extracted_code
          2. generates from source + random_code
          3. loss: l1 between target and images generated with extracted_code
          4. loss: adv of images generated with extracted_code
          5. loss: adv of images generated with random_code

        discriminator:
          1. discriminates images generated with extracted_code
          2. discriminates images generated with random_code
          3. discriminates target images


        A) gen_tape(persistent):
          1. encoder(x) yields extracted_code
          2. generator(x, extracted) yields fake_extracted
          3. generator(x, random   ) yields fake_random
          4. encoder(fake_random) yields recovered_code
          5. discriminator([fake_extracted, fake_random]) yields patches
        g_loss: l1(target-fake_extracted) + adv(patches)
        e_loss: l1(random - recovered_code)

        B) disc_tape(persistent):
          1. generator(x, extracted) yields fake_extracted
          2. generator(x, random   ) yields fake_random
          3. discriminator([fake_extracted, fake_random, target]) yields patches
        d_loss = adv(patches)

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
        half_batch_size = batch_size // 2

        # updates the annealing scheduler to get the new temperature
        temperature = self.annealing_scheduler.update(t)

        # a. determines random domains to be dropped out
        source_images, input_keep_mask = self.sampler.sample(batch, t)
        # source_images (shape=[b, d, s, s, c])
        # input_keep_mask (shape=[b, d])

        visible_source_images = source_images * input_keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # visible_source_images (shape=[b, d, s, s, c])

        # 0. generates random codes and inpaint mask
        noise_length = self.config.noise
        random_codes = tf.random.normal([half_batch_size, noise_length])
        # random_codes (shape=[hb, noise_length])
        masked_source_images, inpaint_mask = self.mask_creator.apply(visible_source_images, t)
        # masked_source_images (shape=[b, d, s, s, c])
        # inpaint_mask (shape=[b, s, s, 1])

        # 1. extracts the palette from the source images
        source_palette = palette_utils.batch_extract_palette_ragged(source_images)
        source_palette = source_palette.to_tensor(default_value=(-1., -1., -1., -1.))
        # source_palette (shape=[b, n, c]) as an ex-ragged, dense tensor

        # 5. prepares the ground truth for the generator
        # for each generator_scale, the target images are the source images downsampled by 1/2
        # (e.g., if generator_scales=3, then the target images have sizes 64x64, then 32x32, then 16x16)
        real_images = []
        for s in range(self.config.generator_scales):
            downscaling_factor = 2 ** s
            target_image = source_images[:half_batch_size]
            target_image = tf.reshape(target_image, (half_batch_size * number_of_domains,
                                                     image_size, image_size, channels))
            target_image = tf.image.resize(target_image, (image_size // downscaling_factor,
                                                          image_size // downscaling_factor),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            target_image = tf.reshape(target_image, (half_batch_size, number_of_domains,
                                                     image_size // downscaling_factor,
                                                     image_size // downscaling_factor, channels))
            real_images.append(target_image)
        # real_images ([generator_scales x shape=[hb, d, ?, ?, c]])

        # A. starts the generator training: first, image->latent->image, then latent->image->latent
        with (tf.GradientTape(persistent=True) as gen_tape):
            # A.1. extracts the latent codes from the source images
            extracted_codes, ec_mean, ec_log_var = self.encode(masked_source_images[:half_batch_size],
                                                               input_keep_mask[:half_batch_size])
            # extracted_codes (shape=[hb, noise_length])

            # A.2-3. generates images using the extracted codes, then with random codes
            generated_images = self.generator(
                self.gen_supplier(masked_source_images, inpaint_mask, source_palette, input_keep_mask,
                                  tf.concat([extracted_codes, random_codes], axis=0)), training=True)
            generated_images = listify(generated_images)
            # generated_images ([generator_scales x shape=[b, d, ?, ?, c]])

            generated_images_with_extracted_codes, generated_images_with_random_codes = keras_utils.scales_output_to_two_halves(
                generated_images)
            # generated_images_with_* ([generator_scales x shape=[hb, d, ?, ?, c]])

            # A.4. extracts the latent codes from the images generated with random codes
            recovered_codes_mean, _ = self.diversity_encoder([generated_images_with_random_codes[0],
                                                              tf.ones_like(input_keep_mask[half_batch_size:])])
            # recovered_codes (shape=[hb, noise_length])

            # A.5. discriminates the full size generated images that used the extracted and random codes, concatenated
            generated_images_full_size = tf.concat([generated_images_with_extracted_codes[0],
                                                    generated_images_with_random_codes[0]], axis=0)
            # generated_images_full_size (shape=[b, d, s, s, c])

            generated_images_per_domain = tf.unstack(generated_images_full_size, axis=1)
            # generated_images_per_domain (d x [b, s, s, c])

            fake_predicted = [listify(self.discriminators[d](generated_images_per_domain[d], training=True))
                              for d in range(number_of_domains)]
            # fake_predicted ([d] x [ds] x shape=[b, ?, ?, 1]) where ds is the number of discriminator scales

            # A.6. calculates the generator loss
            g_loss = self.generator_loss(
                fake_predicted,
                generated_images, real_images,
                input_keep_mask,
                ec_mean, ec_log_var, random_codes, recovered_codes_mean,
                source_palette, temperature,
                batch_shape
            )
            # g_loss (dict with keys representing each loss term)

        # A.7. applies the generator and encoder gradients
        generator_gradients = gen_tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        diversity_encoder_gradients = gen_tape.gradient(g_loss["total"],
                                                        self.diversity_encoder.trainable_variables)
        self.diversity_encoder_optimizer.apply_gradients(zip(diversity_encoder_gradients,
                                                             self.diversity_encoder.trainable_variables))
        del gen_tape

        # B. now we train the discriminators
        # B.1. generates the images from the generator
        generated_images = self.generator(
            self.gen_supplier(masked_source_images, inpaint_mask, source_palette, input_keep_mask,
                              tf.concat([extracted_codes, random_codes], axis=0)), training=True)
        generated_images = listify(generated_images)
        # generated_images ([generator_scales x shape=[b, d, ?, ?, c]])
        # gets only the last output of the generator, excluding the intermediate ones
        generated_images = generated_images[0]
        # generated_images (shape=[b, d, s, s, c])

        with tf.GradientTape(persistent=True) as disc_tape:
            # 3. calculates the discriminator losses
            real_predicted_all_discriminators = []
            fake_predicted_all_discriminators = []
            for i, disc in enumerate(self.discriminators):
                target_domain = i
                real_images = source_images[:, target_domain]
                fake_images = generated_images[:, target_domain]
                real_predicted, fake_predicted = self.mix_and_discriminate(disc, real_images, fake_images,
                                                                           batch_size)
                # real_predicted, fake_predicted ([ds] x [b, ?, ?, 1])
                real_predicted_all_discriminators.append(real_predicted)
                fake_predicted_all_discriminators.append(fake_predicted)

            # xxxx_predicted_all_discriminators ([d] x [ds] x [b, ?, ?, 1])
            d_loss = self.discriminator_loss(real_predicted_all_discriminators, fake_predicted_all_discriminators)

        # 4. calculates the gradients and applies them, then releases the persistent tape
        discriminator_gradients = [disc_tape.gradient(d_loss["total"][d], self.discriminators[d].trainable_variables)
                                   for d in range(number_of_domains)]
        discriminator_gradients = [g for grad in discriminator_gradients for g in grad]
        all_discriminator_trainable_variables = [v for d in range(number_of_domains)
                                                 for v in self.discriminators[d].trainable_variables]
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, all_discriminator_trainable_variables))

        del disc_tape

        summary_step = step // evaluate_steps
        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", g_loss["total"], step=summary_step)
                tf.summary.scalar("adversarial_loss", g_loss["adversarial"], step=summary_step)
                tf.summary.scalar("reconstruction_loss", g_loss["reconstruction"], step=summary_step)
                tf.summary.scalar("latent_loss", g_loss["latent-reconstruction"], step=summary_step)
                tf.summary.scalar("kl_loss", g_loss["kl"], step=summary_step)
                tf.summary.scalar("palette_loss", g_loss["palette-coverage"], step=summary_step)

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                fake_loss = real_loss = weight_loss = total_loss = 0
                for i in range(number_of_domains):
                    fake_loss += d_loss["fake"][i]
                    real_loss += d_loss["real"][i]
                    weight_loss += d_loss["l2-regularization"][i]
                    total_loss += d_loss["total"][i]
                tf.summary.scalar("fake_loss", tf.reduce_mean(fake_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("real_loss", tf.reduce_mean(real_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("weight_loss", tf.reduce_mean(weight_loss / number_of_domains),
                                  step=step // evaluate_steps)
                tf.summary.scalar("total_loss", tf.reduce_mean(total_loss / number_of_domains),
                                  step=step // evaluate_steps)

    def mix_and_discriminate(self, discriminator, real_image, fake_image, half_batch_size):
        """
        Mixes the real and fake images, shuffles them, and then discriminates.
        :param discriminator: the network used for discrimination.
        :param real_image: tensor with real images, shape=[b*d, s, s, c]
        :param fake_image: tensor with fake images, shape=[b*d, s, s, c]
        :param half_batch_size: the size of each half batch (hb).
        :return: discriminated patches for real and fake images. As there are 3 discriminator scales, the output
            has shape [ds] x [b*d, ?, ?, 1] where ds is the number of discriminator scales,
        """
        batch_size = half_batch_size * 2
        discriminator_input = tf.concat([real_image, fake_image], axis=0)

        # create shuffled indices and remember the original positions (through tf.argsort(shuffled))
        shuffled_indices = tf.random.shuffle(tf.range(batch_size))
        discriminator_input = tf.gather(discriminator_input, shuffled_indices)
        inverse_indices = tf.argsort(shuffled_indices)

        # discriminate the shuffled combined half batches
        predicted_patches = discriminator(discriminator_input, training=True)

        # Get the predictions in the original order
        real_predicted_patches = []
        fake_predicted_patches = []
        for ds in range(self.config.discriminator_scales):
            predicted_patches_ds = tf.gather(predicted_patches[ds], inverse_indices)

            # split in real and fake predictions
            real_predicted_patches.append(predicted_patches_ds[:half_batch_size])
            fake_predicted_patches.append(predicted_patches_ds[half_batch_size:])

        # real_predicted_patches ([ds] x shape=[b*d, ?, ?, 1])
        # fake_predicted_patches ([ds] x shape=[b*d, ?, ?, 1])
        return real_predicted_patches, fake_predicted_patches

    def select_examples_for_visualization(self, train_ds, test_ds):
        """
        Selects examples from the training and test datasets for visualization.
        :param train_ds:
        :param test_ds:
        :return: a list of examples, where each example is a tuple containing:
            - source images (shape=[b, d, s, s, c])
            - masked images (shape=[b, d, s, s, c])
            - keep mask (shape=[b, d])
            - inpaint mask (shape=[b, s, s, 1])
            - target palette (shape=[b, (n), c])
        """
        examples = super().select_examples_for_visualization(train_ds, test_ds)
        # examples (list of tuples with: source images, keep mask), [examples] x [d, s, s, c], [examples] x [d]
        # now we add to the examples: inpaint mask and target palette
        number_of_examples_per_partition = len(examples) // 2

        # inpainting masks: from hard to easy, with the last one having a full mask
        number_of_holes = list(range(4, 0, -1))[:number_of_examples_per_partition - 1] + [0]
        example_masks = [create_random_inpaint_mask(tf.stack(examples[ex][0])[tf.newaxis, ...], holes) for ex, holes in
                         enumerate(number_of_holes * 2)]
        # example_masks (list of tuples with: masked source images, inpaint mask)

        # extract the source palettes and make them the target palettes
        source_palettes = [palette_utils.batch_extract_palette_ragged(tf.stack(ex[0])[tf.newaxis, ...])[0]
                           for ex in examples]
        # source_palettes ([examples] x shape=[(n), c])

        # associate the inpaint masks and target palettes with the examples
        examples = [(ex[0], example_masks[i][0], ex[1], example_masks[i][1], source_palettes[i])
                    for i, ex in enumerate(examples)]
        # examples (list of tuples with: source images, masked images, keep mask, inpaint mask, target palette)
        return examples

    def preview_generated_images_during_training(self, examples, save_name, step):
        # cols: masked_source, inpaint_mask, target_images, generated from extracted code, generated from random code
        number_of_domains = self.config.number_of_domains
        image_size = self.config.image_size
        channels = self.config.inner_channels
        domains = self.config.domains_capitalized
        titles = ["Source Images", "Inpaint Mask", "Target Images", "Gen. Extr. Code", "Gen. Rnd Code"]
        num_rows = len(examples)
        num_cols = len(titles)

        if step is not None:
            if step == 1:
                step = 0
            for d in range(1, number_of_domains + 1):
                titles[-1 * d] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        source_images, masked_images, keep_mask, inpaint_mask, source_palettes = zip(*examples)
        source_images = tf.stack(source_images, axis=0)
        masked_images = tf.concat(masked_images, axis=0)
        keep_mask = tf.stack(keep_mask, axis=0)
        inpaint_mask = tf.concat(inpaint_mask, axis=0)
        source_palettes = keras_utils.list_of_palettes_to_ragged_tensor(source_palettes)
        source_palettes = source_palettes.to_tensor(default_value=(-1., -1., -1., -1.))

        visible_masked_images = masked_images * keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        extracted_codes, _, _ = self.encode(source_images, keep_mask, training=True)
        random_codes = tf.random.normal([num_rows, self.config.noise])

        generated_images_with_extracted_codes = self.generator(self.gen_supplier(
            visible_masked_images,
            inpaint_mask,
            source_palettes,
            keep_mask,
            extracted_codes), training=True)
        # generated_images_with_extracted_codes (gs x shape=[b, d, s, s, c]) or (shape=[b, d, s, s, c]) if gs=1)
        generated_images_with_extracted_codes = listify(generated_images_with_extracted_codes)[0]
        # generated_images_with_extracted_codes ([gs] x shape=[b, d, s, s, c])

        generated_images_with_random_codes = self.generator(self.gen_supplier(
            visible_masked_images,
            inpaint_mask,
            source_palettes,
            keep_mask,
            random_codes), training=True)
        generated_images_with_random_codes = listify(generated_images_with_random_codes)[0]
        # generated_images_with_random_codes ([gs] x shape=[b, d, s, s, c])

        column_contents = [visible_masked_images, inpaint_mask, source_images,
                           generated_images_with_extracted_codes, generated_images_with_random_codes]

        # use matplotlib to plot the images with the following structure:
        # - each row is an example
        # - each column is a different image type that uses column_contents[col]. If there rank is 5, each cell in
        # that column should contain a 2x2 grid of images with shape [s, s, c] each
        fig = plt.figure(figsize=(8 * num_cols, 8 * num_rows))
        sub_figs = fig.subfigures(num_rows, num_cols)
        for row in range(num_rows):
            for col in range(num_cols):
                content = column_contents[col][row]
                # content (shape=[d, s, s, c]) or (shape=[s, s, c]) or (shape=[s, s, 1])
                sub_fig = sub_figs[row, col]
                if row == 0:
                    sub_fig.suptitle(titles[col], fontsize=36)

                if content.shape.rank == 4:
                    # 2x2 subplots, content (shape=[d, s, s, c])
                    axes = sub_fig.subplots(2, 2)
                    for i_d in range(2):
                        for j_d in range(2):
                            ax = axes[i_d, j_d]
                            ax.imshow(tf.clip_by_value(content[i_d * 2 + j_d] * 0.5 + 0.5, 0., 1.))
                            ax.axis("off")
                elif content.shape.rank == 3:
                    # single image, content (shape=[s, s, c]) or (shape=[s, s, 1])
                    ax = sub_fig.subplots(1, 1)
                    if content.shape[-1] == 1:
                        ax.imshow(content[:, :, 0], cmap="gray")
                    else:
                        ax.imshow(tf.clip_by_value(content * 0.5 + 0.5, 0., 1.))
                    ax.axis("off")

        figure.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        return figure

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        batch_size = self.config.batch

        def generate_images_from_example_indices(example_indices):
            domain_images, keep_mask, possible_target_domain = example_indices
            # domain_images (b, d, s, s, c)
            # keep_mask (b, d, 1, 1, 1)
            images_shape = tf.shape(domain_images)
            num_examples, number_of_domains, image_size = images_shape[0], images_shape[1], images_shape[2]

            visible_source_images = domain_images * keep_mask
            # visible_source_images (b, d, s, s, c)
            source_palettes = palette_utils.batch_extract_palette_ragged(domain_images)
            source_palettes = source_palettes.to_tensor(default_value=(-1., -1., -1., -1.))

            keep_mask_oh = tf.reshape(keep_mask, [num_examples, number_of_domains])
            extracted_codes_mean, _ = self.diversity_encoder.predict([visible_source_images, keep_mask_oh],
                                                                     verbose=0,
                                                                     batch_size=batch_size)
            generated_images = self.generator.predict(self.gen_supplier(
                visible_source_images,
                tf.zeros((num_examples, image_size, image_size, 1)),
                source_palettes,
                keep_mask_oh,
                extracted_codes_mean),
                verbose=0,
                batch_size=batch_size
            )
            generated_images = listify(generated_images)[0]
            # generated_images (shape=[b, d, s, s, c])

            fake_images = tf.gather(generated_images, possible_target_domain, batch_dims=1)
            real_images = tf.gather(domain_images, possible_target_domain, batch_dims=1)
            return real_images, fake_images

        return {
            "train": generate_images_from_example_indices(example_indices_for_evaluation["train"]),
            "test": generate_images_from_example_indices(example_indices_for_evaluation["test"])
        }

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        """
        Generates an images from the dataset for visualization purposes in the end of training.
        *WARNING*: assumes 4 domains: back, left, right, front when selecting which keep_masks to use.
        :param enumerated_dataset:
        :param step:
        :param num_images:
        :return:
        """
        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        image_size = self.config.image_size
        noise_length = self.config.noise
        batch_size = self.config.batch
        channels = self.config.inner_channels

        # WARNING: assumes fixed 4 domains. There would be 15 rows if we wanted to showcase all combinations of inputs
        # num_cols = 14, num_rows = 6
        keep_mask = [[1, 1, 1, 1],  # all 4 images
                     [1, 1, 0, 1],  # missing FRONT
                     [1, 0, 1, 0],  # BACK + FRONT
                     [0, 0, 1, 1],  # FRONT + RIGHT
                     [0, 0, 1, 0],  # FRONT only
                     [0, 0, 0, 1]  # RIGHT only
                     ]
        keep_mask = tf.constant(keep_mask, dtype=tf.float32)
        # keep_mask (shape=(num_rows, d))
        num_cols = number_of_domains * 3 + 2
        num_rows = keep_mask.shape[0]

        # inpaint_mask is blank for the first 2 groups of columns and random for the third
        blank_inpaint_mask = tf.zeros((num_rows, image_size, image_size, 1))

        # blank_inpaint_mask (shape=[num_rows, s, s, 1])

        # the inputs to the generator are:
        # - masked source images (shape=[b, d, s, s, c])
        # - inpaint mask (shape=[b, s, s, 1])
        # - source palettes (shape=[b, (n), c])
        # - keep mask (shape=[b, d])
        # - extracted codes (shape=[b, noise_length])

        # we want to generate an image for each character in the dataset. There are 14 columns and 8 rows:
        # - columns:  1. source_images, 2-5. images generated without an inpaint mask with the extracted
        #   style code for each target domain: back, left, right, front, 6-9. images generated without an inpaint mask
        #   with a random style code for each target domain, 10. inpaint mask, 11-14. images generated with
        #   the original, extracted style code and using an inpaint mask for all target domains
        # - rows: each row will have a different permutation of domains to drop, so that 1. all inputs, 2. 3 inputs,
        #   3-5. 2 inputs, 6-8. 1 input

        # preparing the inputs for the generator:
        # - the batch size will be 24: 8 rows x 3 groups of columns
        # - the inpaint mask is an empty one for the first 2 groups of columns and a random for the third, for each row
        # - the source palettes are extracted from the visible source images
        # - the keep mask is a multi-hot encoded vector indicating which domains are available for each row
        # - the codes are extracted from the visible source images and used for the column groups 1 and 3, and a random
        #   code is used for the column group 2

        def generate_images_for_example(domain_images, idx):
            """
            Generates and plots an image for a single example.
            :param domain_images: the source images for the example, shape=[d, s, s, c]
            :param idx: the index of the example in the dataset.
            """
            # 1. repeat the domain_images to match the number of rows
            domain_images = tf.repeat(tf.expand_dims(domain_images, 0), num_rows, axis=0)
            # domain_images (shape=[num_rows, d, s, s, c])
            visible_source_images = domain_images * keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
            # visible_source_images (shape=[num_rows, d, s, s, c])

            # 2. extracts the source palettes
            source_palettes = palette_utils.batch_extract_palette_ragged(visible_source_images)
            source_palettes = source_palettes.to_tensor(default_value=(-1., -1., -1., -1.))
            # source_palettes (shape=[num_rows, max_colors, c])

            # 3. creates the inpaint mask for the third group of columns
            masked_source_images, random_inpaint_masks = keras_utils.create_random_inpaint_mask(
                visible_source_images, 2)
            # masked_source_images (shape=[num_rows, d, s, s, c])
            # random_inpaint_masks (shape=[num_rows, s, s, 1])

            # 4. extracts the codes from the visible source images
            # keep_mask_oh = tf.reshape(keep_mask, [num_rows, number_of_domains])
            extracted_codes_mean, _ = self.diversity_encoder.predict([visible_source_images, keep_mask], verbose=0)
            random_codes = tf.random.normal((num_rows, noise_length))
            # extracted_codes_mean, random_codes (shape=[num_rows, noise_length])

            # 5. generates the images using the generator
            generated_images = self.generator.predict(self.gen_supplier(
                # source_images: (num_rows x 3 x shape=[d, s, s, c]))
                tf.concat([visible_source_images, visible_source_images, masked_source_images], axis=0),
                # inpaint_mask: (num_rows x 3 x shape=[s, s, 1])
                tf.concat([blank_inpaint_mask, blank_inpaint_mask, random_inpaint_masks], axis=0),
                # source_palettes: (num_rows x 3 x shape=[(n), c])
                tf.repeat(source_palettes, 3, axis=0),
                # keep_mask: (num_rows x 3 x shape=[d])
                tf.repeat(keep_mask, 3, axis=0),
                # codes: (num_rows x 3 x shape=[noise_length])
                tf.concat([extracted_codes_mean, random_codes, extracted_codes_mean], axis=0)
            ), verbose=0, batch_size=batch_size)
            generated_images = listify(generated_images)[0]
            # generated_images (shape=[num_rows x 3, d, s, s, c])

            # 6. reshapes the generated images to have the shape [num_rows, 3, d, s, s, c]
            generated_images = tf.reshape(generated_images, (num_rows, 3, number_of_domains,
                                                             image_size, image_size, channels))
            # generated_images (shape=[num_rows, 3, d, s, s, c])

            # 7. plots the images in a grid format with 14 columns and 6 rows
            # - the first column is the source images in a 2x2 subgrid
            # - columns 2-5 are the images generated with the extracted style code for each target domain
            # - columns 6-9 are the images generated with a random style code for each target domain
            # - column 10 is the inpaint mask
            # - columns 11-14 are the images generated with the extracted style code and using an inpaint mask for all
            #   target domains
            fig = plt.figure(figsize=(8 * num_cols, 8 * num_rows), layout="constrained")
            sub_figs = fig.subfigures(num_rows, num_cols)
            for row in range(num_rows):
                # 1. source images
                sub_fig = sub_figs[row, 0]
                ax = sub_fig.subplots(2, 2)
                for i_d in range(2):
                    for j_d in range(2):
                        ax[i_d, j_d].imshow(
                            tf.clip_by_value(visible_source_images[row, i_d * 2 + j_d] * 0.5 + 0.5, 0., 1.))
                        ax[i_d, j_d].axis("off")
                if row == 0:
                    sub_fig.suptitle(f"Source", fontsize=36)

                # 2-5. generated images with extracted style code
                for col in range(1, 5):
                    sub_fig = sub_figs[row, col]
                    ax = sub_fig.subplots(1, 1)
                    ax.imshow(tf.clip_by_value(generated_images[row, 0, col - 1] * 0.5 + 0.5, 0., 1.))
                    ax.axis("off")
                    domain_name = self.config.domains_capitalized[col - 1]
                    if row == 0:
                        sub_fig.suptitle(f"Gen. Ori. {domain_name}", fontsize=36)

                # 6-9. generated images with random style code
                for col in range(5, 9):
                    sub_fig = sub_figs[row, col]
                    ax = sub_fig.subplots(1, 1)
                    ax.imshow(tf.clip_by_value(generated_images[row, 1, col - 5] * 0.5 + 0.5, 0., 1.))
                    ax.axis("off")
                    domain_name = self.config.domains_capitalized[col - 5]
                    if row == 0:
                        sub_fig.suptitle(f"Gen. Rnd. {domain_name}", fontsize=36)

                # 10. inpaint mask
                sub_fig = sub_figs[row, 9]
                ax = sub_fig.subplots(1, 1)
                ax.imshow(random_inpaint_masks[row], cmap="gray")
                ax.axis("off")
                if row == 0:
                    sub_fig.suptitle("Inpaint Mask", fontsize=36)

                # 11-14. generated images with extracted style code and inpaint mask
                for col in range(10, num_cols):
                    sub_fig = sub_figs[row, col]
                    ax = sub_fig.subplots(1, 1)
                    ax.imshow(tf.clip_by_value(generated_images[row, 2, col - 10] * 0.5 + 0.5, 0., 1.))
                    ax.axis("off")
                    domain_name = self.config.domains_capitalized[col - 10]
                    if row == 0:
                        sub_fig.suptitle(f"Inpainted Ori. {domain_name}", fontsize=36)

            # fig.tight_layout()
            image_path = os.sep.join([base_image_path, f"{idx:04d}_at_step_{step}.png"])
            plt.savefig(image_path, transparent=True)
            plt.close(fig)

        for idx, domain_images in tqdm(enumerated_dataset, total=num_images):
            generate_images_for_example(domain_images, idx)

    def debug_discriminator_output(self, batch, image_path):
        d_scales = self.config.discriminator_scales
        g_scales = self.config.generator_scales
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
            visible_source_images = batch[i] * keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
            visible_source_images = tf.expand_dims(visible_source_images, axis=0)
            # visible_source_images (1, d, s, s, c)

            extracted_code = self.diversity_encoder.predict([visible_source_images, keep_mask[tf.newaxis, ...]],
                                                            verbose=0)[0]
            # extracted_code (shape=[1, noise_length])

            masked_source_images, inpaint_mask = keras_utils.create_random_inpaint_mask(visible_source_images, 2)
            # masked_source_images (shape=[1, d, s, s, c])
            # inpaint_mask (shape=[1, s, s, 1])
            source_palette = palette_utils.batch_extract_palette_ragged(visible_source_images)
            source_palette = source_palette.to_tensor(default_value=(-1., -1., -1., -1.))
            # source_palette (shape=[1, n, c])

            fake_image = self.generator.predict(
                self.gen_supplier(
                    masked_source_images,
                    inpaint_mask,
                    source_palette,
                    keep_mask[tf.newaxis, ...],
                    extracted_code
                ), verbose=0)
            fake_image = listify(fake_image)
            fake_images.append(fake_image)
        # fake_images (list of [b] x [gs] x shape=[1, d, ?, ?, c]])

        # gets the result of discriminating the real and fake (translated) images
        real_patches = [listify(self.discriminators[target_domains[i]](real_images[i][tf.newaxis, ...]))
                        for i in range(batch_size)]
        fake_patches = [listify(self.discriminators[target_domains[i]](fake_images[i][0][0][target_domains[i]][tf.newaxis, ...]))
                        for i in range(batch_size)]
        # if d_scales == 1:
        #     real_patches = [[real_patches[i]] for i in range(batch_size)]
        #     fake_patches = [[fake_patches[i]] for i in range(batch_size)]
        # [b] x [ds] x shape=[1, x, x, 1]

        # calculates the mean of the patches for each discriminator scale
        real_means = [0. for _ in range(d_scales)]
        fake_means = [0. for _ in range(d_scales)]
        for ds in range(d_scales):
            for b in range(batch_size):
                real_means[ds] += tf.reduce_mean(real_patches[b][ds])
                fake_means[ds] += tf.reduce_mean(fake_patches[b][ds])
            real_means[ds] /= tf.cast(batch_size, tf.float32)
            fake_means[ds] /= tf.cast(batch_size, tf.float32)

        # lsgan yields an unbounded real number, which should be 1 for real images and 0 for fake
        # but, we need to provide them in the [0, 1] range
        flattened_real_patches = tf.concat([tf.squeeze(tf.reshape(real_patches[i][c], [-1])) for i in range(batch_size)
                                            for c in range(d_scales)], axis=0)
        flattened_fake_patches = tf.concat([tf.squeeze(tf.reshape(fake_patches[i][c], [-1])) for i in range(batch_size)
                                            for c in range(d_scales)], axis=0)
        concatenated_predictions = tf.concat([flattened_real_patches, flattened_fake_patches], axis=0)
        min_value = tf.reduce_min(concatenated_predictions)
        max_value = tf.reduce_max(concatenated_predictions)
        amplitude = max_value - min_value
        real_predicted = [[(real_patches[i][c] - min_value) / amplitude for c in range(d_scales)]
                          for i in range(batch_size)]
        fake_predicted = [[(fake_patches[i][c] - min_value) / amplitude for c in range(d_scales)]
                          for i in range(batch_size)]

        discriminator_titles = [f"Disc. Scale {c}" for c in range(d_scales)]
        generator_titles = [f"Imputed (scale {x})" for x in range(g_scales)]
        titles = ["Real", *discriminator_titles, *generator_titles, *discriminator_titles]
        num_cols = len(titles)
        num_rows = batch_size.numpy()

        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
        is_discriminator_scales_column = lambda col: 0 < col <= d_scales or d_scales + g_scales < col
        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows, num_cols, (i * num_cols) + j + 1)
                subplot_title = ""
                if i == 0:
                    if is_discriminator_scales_column(j):
                        # it's a discriminator scale column... append the mean of the patches to the title
                        means = real_means if j <= d_scales else fake_means
                        subtractor = 1 if j <= d_scales else d_scales + g_scales + 1
                        titles[j] += f" ({means[j - subtractor]:.3f})"
                    subplot_title = titles[j]
                plt.title(subplot_title, fontdict={"fontsize": 24})

                imshow_args = {}
                if j == 0:
                    # the real image
                    image = real_images[i] * 0.5 + 0.5
                elif 0 < j < d_scales + 1:
                    # the discriminated real image in one of the scales
                    image = tf.squeeze(real_predicted[i][j - 1])
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                elif j <= d_scales + g_scales:
                    # the imputed image or some reduced scale
                    subtractor = d_scales + 1
                    # fake_images (list of [batch x generator_scales x shape=[1, d, ?, ?, c]])
                    image = tf.clip_by_value(fake_images[i][j-subtractor][0][target_domains[i]] * 0.5 + 0.5, 0., 1.)
                elif j > d_scales + g_scales:
                    # the discriminated fake image in one of the scales
                    image = tf.squeeze(fake_predicted[i][j - d_scales - g_scales - 1])
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                else:
                    raise ValueError(f"Invalid column index {j}")
                plt.axis("off")
                plt.imshow(image, **imshow_args)

        fig.tight_layout()
        plt.savefig(image_path, transparent=True)
        plt.close(fig)


from tensorflow.keras import layers, models


class KLRegularizer(layers.Layer):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def call(self, inputs):
        mu, logvar = tf.split(inputs, 2, axis=-1)
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        self.add_loss(self.weight * kl_loss)
        return inputs


class FiLMLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        self.gamma = layers.Dense(input_shape[0][-1], use_bias=False)
        self.beta = layers.Dense(input_shape[0][-1], use_bias=False)

    def call(self, inputs):
        x, condition = inputs
        gamma = self.gamma(condition)[:, None, None, :]
        beta = self.beta(condition)[:, None, None, :]
        return gamma * x + beta


def build_monolith_generator(config):
    domains = config.number_of_domains
    image_size = config.image_size
    channels = config.inner_channels
    noise_length = config.noise
    film_length = config.film
    generator_scales = config.generator_scales
    quantize_to_palette = config.palette_quantization

    # define the inputs
    masked_images_input = layers.Input(shape=(domains, image_size, image_size, channels), name="source_images")
    inpaint_mask_input = layers.Input(shape=(image_size, image_size, 1), name="inpaint_mask")
    target_palette_input = layers.Input(shape=(None, channels), name="desired_palette")
    domain_availability_input = layers.Input(shape=(domains,), name="domain_availability")
    noise_input = layers.Input(shape=(noise_length,), name="noise")

    # processes the domain availability and the noise to generate embeddings
    # they are used to condition the generator
    domain_embedding = layers.Dense(film_length)(domain_availability_input)
    noise_embedding = layers.Dense(film_length)(noise_input)

    # mix the source images with the inpaint mask
    # reshape the inpaint_mask_input to allow concatenation
    x = keras_utils.ConcatenateMask()([masked_images_input, inpaint_mask_input])
    # x (shape=[b, d, s, s, c + 1])
    x = layers.Reshape((image_size, image_size, domains * (channels + 1)))(x)

    number_of_filters = [32, 64, 128, 256]
    init = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(number_of_filters[0], kernel_size=4, strides=1, padding="same", kernel_initializer=init)(x)
    x = layers.GroupNormalization(groups=number_of_filters[0], epsilon=0.00001)(x)
    x = layers.LeakyReLU()(x)

    # encoder blocks
    skip_connections = []
    for filters in number_of_filters:
        x = resblock(x, filters, 4, init)
        skip_connections += [x]

        # downsample
        x = layers.Conv2D(filters * 2, kernel_size=4, strides=2, padding="same", kernel_initializer=init,
                          use_bias=False)(x)
        x = layers.GroupNormalization(groups=filters, epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    # bottleneck
    x = resblock(x, number_of_filters[-1] * 2, 4, init)
    x = layers.ReLU()(x)

    # decoder blocks with intermediate outputs and skip connections
    outputs = []
    number_of_filters.reverse()
    for i, filters in enumerate(number_of_filters):
        # upsample
        x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init,
                                   use_bias=False)(x)
        x = layers.ReLU()(x)
        x = FiLMLayer()([x, domain_embedding])
        x = FiLMLayer()([x, noise_embedding])
        x = layers.Concatenate()([x, skip_connections.pop()])

        x = resblock(x, filters * 2, 4, init)
        x = layers.ReLU()(x)

        if generator_scales > 1:
            reverse_i = len(number_of_filters) - i
            if generator_scales >= reverse_i > 1:
                # add intermediate output
                intermediate_output = layers.Conv2D(domains * channels, kernel_size=2, padding="same",
                                                    kernel_initializer=init, activation="tanh")(x)
                current_size = image_size // (2 ** (reverse_i - 1))
                output_layer_name = f"intermediate-output-{current_size}x{current_size}"
                intermediate_output = layers.Reshape((domains, current_size, current_size, channels),
                                                     name=output_layer_name)(intermediate_output)
                outputs.append(intermediate_output)

    pre_output = layers.Conv2D(domains * channels, kernel_size=4, padding="same", kernel_initializer=init)(x)
    pre_output = layers.Activation("tanh")(pre_output)

    if quantize_to_palette:
        # quantize to the palette
        # change the dimensions so the domains come first, then height, width and channels
        pre_output = layers.Reshape((domains, image_size, image_size, channels),
                                    name=f"pre-quantization-{image_size}x{image_size}")(pre_output)
        quantization_layer = keras_utils.DifferentiablePaletteQuantization(name="quantized-images")
        final_output = quantization_layer([pre_output, target_palette_input])
        inputs = [masked_images_input, inpaint_mask_input, target_palette_input, domain_availability_input, noise_input]
    else:
        # change the dimensions so the domains come first, then height, width and channels
        final_output = layers.Reshape((domains, image_size, image_size, channels),
                                      name=f"output-images")(pre_output)
        inputs = [masked_images_input, inpaint_mask_input, domain_availability_input, noise_input]

    outputs.append(final_output)
    outputs.reverse()

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"SpriteMonolithGenerator{'_Quantized' if quantize_to_palette else ''}"
    )

    if quantize_to_palette:
        model.quantization = quantization_layer

    return model


def build_munitlike_discriminator(domain_letter, image_size, channels, scales):
    return munit_discriminator_multi_scale(domain_letter, image_size, channels, scales)


def build_diversity_encoder(config):
    """
    Builds the diversity encoder for the Sprite Editor model.
    This encoder is used to extract latent codes from the source images.
    :param config: configuration object containing model parameters.
    :return: a Keras Model representing the diversity encoder.
    """
    image_size = config.image_size
    channels = config.inner_channels
    noise_length = config.noise
    domains = config.number_of_domains

    # define the inputs
    source_images_input = layers.Input(shape=(domains, image_size, image_size, channels), name="source_images")
    domain_availability_input = layers.Input(shape=(domains,), name="domain_availability")

    # multi-hot encode and spread on height/width to match the image size
    domain_availability = layers.CategoryEncoding(num_tokens=domains, output_mode="multi_hot")(
        domain_availability_input)
    domain_availability = keras_utils.TileLayer(image_size)(domain_availability)
    domain_availability = keras_utils.TileLayer(image_size)(domain_availability)

    # concat the images with the domain availability
    source_images = layers.Reshape((image_size, image_size, domains * channels))(source_images_input)
    x = layers.Concatenate(axis=-1)([source_images, domain_availability])

    # encoder blocks
    init = tf.random_normal_initializer(0., 0.02)
    number_of_filters = [32, 64, 128]
    # feature_map_size = image_size

    x = layers.Conv2D(number_of_filters[0], kernel_size=4, strides=1, padding="same", kernel_initializer=init)(x)
    x = layers.GroupNormalization(groups=number_of_filters[0], epsilon=0.00001)(x)
    x = layers.LeakyReLU()(x)

    for filters in number_of_filters:
        x = resblock(x, filters, 4, init)
        # downsample
        x = layers.Conv2D(filters * 2, kernel_size=4, strides=2, padding="same", kernel_initializer=init,
                          use_bias=False)(x)
        x = layers.GroupNormalization(groups=filters, epsilon=0.00001)(x)
        x = layers.LeakyReLU()(x)

        # feature_map_size //= 2

    x = layers.Flatten()(x)
    mean_output = layers.Dense(noise_length, kernel_initializer=init)(x)
    variance_output = layers.Dense(noise_length, kernel_initializer=init)(x)

    return models.Model(inputs=[source_images_input, domain_availability_input],
                        outputs=[mean_output, variance_output],
                        name="DiversityEncoder"
                        )

# started to implement with separate encoders, but did not finish it
# def build_domain_encoder(config, identity):
#     image_size = config.image_size
#     channels = config.inner_channels
#     noise_length = config.noise
# 
#     masked_images_input = layers.Input(shape=(image_size, image_size, channels), name="source_images")
#     mask_input = layers.Input(shape=(image_size, image_size, 1), name="mask")
#     target_palette_input = layers.Input(shape=(image_size, image_size, channels), name="desired_palette")
#     domain_availability_input = layers.Input(shape=(config.number_of_domains,), name="domain_availability")
#     noise_input = layers.Input(shape=(noise_length,), name="noise")
# 
#     # ...
# 
#     return models.Model(inputs=[masked_images_input, mask_input, target_palette_input, domain_availability_input,
#                                 noise_input],
#                         outputs=None,  # Replace with actual output layers
#                         name=f"domain_encoder_{identity.lower()}"
#                         )


# python train.py sprite --rm2k --steps 100 --evaluate-steps 100 --vram 4096 --temperature 0.1 --annealing linear
