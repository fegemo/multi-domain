import os
from abc import ABC, abstractmethod

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility import palette_utils, dataset_utils, io_utils, histogram_utils
from .networks import (collagan_affluent_generator, collagan_original_discriminator,
                       collagan_palette_conditioned_with_transformer_generator)
from .side2side_model import S2SModel
from utility.keras_utils import NParamsSupplier, LinearAnnealingScheduler, NoopAnnealingScheduler


class CollaGANModel(S2SModel):
    def __init__(self, config):
        super().__init__(config, export_additional_training_endpoint=True)

        self.lambda_l1 = config.lambda_l1 or config.lambda_l1
        self.lambda_l1_backward = config.lambda_l1_backward or config.lambda_l1
        self.lambda_domain = config.lambda_domain
        self.lambda_ssim = config.lambda_ssim
        self.lambda_palette = config.lambda_palette
        self.lambda_histogram = config.lambda_histogram
        self.lambda_regularization = config.lambda_regularization
        self.lambda_adversarial = config.lambda_adversarial

        if config.input_dropout == "none":
            self.sampler = SimpleSampler(config)
        elif config.input_dropout == "original":
            self.sampler = InputDropoutSampler(config)
        elif config.input_dropout == "aggressive":
            self.sampler = AggressiveInputDropoutSampler(config)
        elif config.input_dropout == "balanced":
            self.sampler = BalancedInputDropoutSampler(config)
        elif config.input_dropout == "conservative":
            self.sampler = ConservativeInputDropoutSampler(config)
        elif config.input_dropout == "curriculum":
            self.sampler = CurriculumLearningSampler(config)
        else:
            raise ValueError(f"The provided {config.input_dropout} type for input dropout has not been implemented.")

        if config.cycled_source_replacer in ["", "dropout"]:
            self.cycled_source_replacer = DroppedOutCycledSourceReplacer(config)
        else:
            self.cycled_source_replacer = ForwardOnlyCycledSourceReplacer(config)

        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.gen_supplier = NParamsSupplier(3 if config.palette_quantization else 2)
        self.generator = self.inference_networks["generator"]
        self.discriminator = self.training_only_networks["discriminator"]

        if config.palette_quantization and config.annealing != "none":
            self.annealing_scheduler = LinearAnnealingScheduler(config.temperature, [self.generator.quantization])
        else:
            self.annealing_scheduler = NoopAnnealingScheduler()

    def create_inference_networks(self):
        config = self.config
        if config.generator in ["colla", "affluent", ""]:
            return {
                "generator": collagan_affluent_generator(config.number_of_domains, config.image_size,
                                                         config.output_channels,
                                                         config.capacity,
                                                         config.palette_quantization)
            }
        elif config.generator in ["palette-transformer"]:
            return {
                "generator": collagan_palette_conditioned_with_transformer_generator(
                    config.number_of_domains, config.image_size,
                    config.output_channels,
                    config.capacity)
            }
        else:
            raise ValueError(f"The provided {config.generator} type for generator has not been implemented.")

    def create_training_only_networks(self):
        config = self.config
        if config.discriminator in ["colla", ""]:
            return {
                "discriminator": collagan_original_discriminator(config.number_of_domains, config.image_size,
                                                                 config.output_channels)
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type for discriminator has not been implemented.")

    def generator_loss(self, fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                       cycled_images, source_images_5d,
                       fake_predicted_domain, cycle_predicted_domain, target_domain,
                       input_dropout_mask, batch_shape, fw_target_palette, bw_target_palette, temperature):
        # cycled_images (shape=[b*d, s, s, c])
        # input_dropout_mask (shape=[b, d])
        number_of_domains = batch_shape[0]
        batch_size, image_size, channels = batch_shape[1], batch_shape[2], batch_shape[4]
        number_of_domains_float = tf.cast(number_of_domains, tf.float32)

        # the source_images need to have the forward target images excluded so it can be properly compared
        # against the cycled_images
        bw_target_domain = tf.tile(tf.range(number_of_domains)[tf.newaxis, ...], [batch_size, 1])
        eliminate_fw_target_mask = tf.not_equal(bw_target_domain, target_domain[..., tf.newaxis])
        bw_target_domain = tf.reshape(tf.boolean_mask(bw_target_domain, eliminate_fw_target_mask),
                                      [batch_size, number_of_domains - 1])

        source_images_5d = tf.gather(source_images_5d, bw_target_domain, batch_dims=1)
        source_images = tf.reshape(source_images_5d,
                                   [batch_size * (number_of_domains - 1), image_size, image_size, channels])
        cycled_images_5d = tf.reshape(cycled_images,
                                      [batch_size, number_of_domains - 1, image_size, image_size, channels])

        input_dropout_mask = tf.gather(input_dropout_mask, bw_target_domain, batch_dims=1)
        input_dropout_mask_1d = tf.reshape(input_dropout_mask, [batch_size * (number_of_domains - 1)])
        # source_images (shape=[b, d, s, s, c])
        # cycle_images_5d (shape=[b, d, s, s, c])
        # input_dropout_mask_1d (shape=[b*d])

        # adversarial (lsgan) loss
        adversarial_forward__loss = tf.reduce_mean(tf.math.squared_difference(fake_predicted_patches, 1.))
        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(cycled_predicted_patches, 1.)) * \
                                    (number_of_domains_float - 1.)
        adversarial_loss = (adversarial_forward__loss + adversarial_backward_loss) / number_of_domains_float

        # l1 (forward, backward)
        l1_forward__loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        # l1_backward_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         # mean of pixel l1s per image, but 0 for dropped out input images
        #         tf.reduce_mean(tf.abs(source_images_5d - cycled_images_5d), axis=[2, 3, 4]) * input_dropout_mask,
        #         axis=1) * tf.reduce_sum(input_dropout_mask, axis=1),
        #     axis=0)
        source_images_5d_kept = tf.boolean_mask(source_images_5d, tf.cast(input_dropout_mask, tf.int32))
        cycled_images_5d_kept = tf.boolean_mask(cycled_images_5d, tf.cast(input_dropout_mask, tf.int32))
        l1_backward_loss = tf.reduce_mean(tf.abs(source_images_5d_kept - cycled_images_5d_kept)) * \
                           tf.reduce_sum(input_dropout_mask)

        # ssim loss (forward, backward)
        ssim_forward_ = tf.image.ssim(fake_image + 1., real_image + 1., 2)
        ssim_backward = tf.image.ssim(cycled_images + 1., source_images + 1., 2) * input_dropout_mask_1d
        # ssim_forward_ (shape=[b,])
        # ssim_backward (shape=[b*d,])
        ssim_forward__loss = tf.reduce_mean(-tf.math.log((1. + ssim_forward_) / 2.))
        # ssim_forward_loss (shape=[b,])
        ssim_backward_loss = tf.reduce_mean(-tf.math.log((1. + ssim_backward) / 2.))
        # ssim_backward_loss (shape=[b,])
        ssim_loss = (ssim_forward__loss + ssim_backward_loss * (number_of_domains_float - 1)) / number_of_domains_float

        # domain classification loss (forward, backward)
        forward__domain = tf.one_hot(target_domain, number_of_domains)
        backward_domain = tf.reshape(tf.one_hot(bw_target_domain, number_of_domains),
                                     [batch_size * (number_of_domains - 1), number_of_domains])
        backward_predicted_domain = cycle_predicted_domain
        # forward__domain (shape=[b, d])
        # backward_domain (shape=[b*d, d])

        classification_forward__loss = self.cce(forward__domain, fake_predicted_domain)
        classification_backward_loss = self.cce(backward_domain, backward_predicted_domain) * \
                                       (number_of_domains_float - 1.)
        classification_loss = (classification_forward__loss + classification_backward_loss) / number_of_domains_float

        # palette loss (forward, backward)
        palette_forward_ = palette_utils.calculate_palette_coverage_loss_ragged(fake_image, fw_target_palette,
                                                                                temperature)
        palette_backward = palette_utils.calculate_palette_coverage_loss_ragged(cycled_images,
                                                                                tf.tile(bw_target_palette,
                                                                                        [number_of_domains - 1, 1, 1]),
                                                                                temperature)
        palette_loss = palette_forward_ + palette_backward

        # histogram loss (forward)
        real_histogram = histogram_utils.calculate_rgbuv_histogram(real_image)
        fake_histogram = histogram_utils.calculate_rgbuv_histogram(fake_image)
        histogram_loss = histogram_utils.hellinger_loss(real_histogram, fake_histogram)

        # regularization loss (l2 - weight decay)
        regularization_loss = tf.reduce_sum(self.generator.losses)

        # observation: ssim loss uses only the backward (cycled) images... that's on the colla's code and paper
        total_loss = self.lambda_adversarial * adversarial_loss + \
                     self.lambda_l1 * l1_forward__loss + self.lambda_l1_backward * l1_backward_loss + \
                     self.lambda_ssim * ssim_backward_loss + \
                     self.lambda_domain * classification_loss + \
                     self.lambda_palette * palette_loss + \
                     self.lambda_histogram * histogram_loss + \
                     self.lambda_regularization * regularization_loss

        return {"total": total_loss, "adversarial": adversarial_loss, "l1_forward": l1_forward__loss,
                "l1_backward": l1_backward_loss, "ssim": ssim_loss, "domain": classification_loss,
                "palette": palette_loss, "histogram": histogram_loss, "weight_decay": regularization_loss}

    def discriminator_loss(self, source_predicted_patches, cycled_predicted_patches, source_predicted_domain,
                           real_predicted_patches, fake_predicted_patches, batch_shape):
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        adversarial_real = tf.reduce_mean(tf.math.squared_difference(real_predicted_patches, 1.))
        adversarial_fake = tf.reduce_mean(tf.math.square(fake_predicted_patches))
        adversarial_forward_loss = adversarial_real + adversarial_fake

        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(source_predicted_patches, 1.)) + \
                                    tf.reduce_mean(tf.math.square(cycled_predicted_patches))
        adversarial_backward_loss *= tf.cast(number_of_domains, tf.float32)

        adversarial_loss = (adversarial_forward_loss + adversarial_backward_loss) / \
                           (tf.cast(number_of_domains, tf.float32) + 1.)

        source_label_domain = tf.tile(tf.one_hot(tf.range(number_of_domains), number_of_domains), [batch_size, 1])
        # source_label_domain (shape=[b*d, d])
        domain_loss = self.cce(source_label_domain, source_predicted_domain)

        total_loss = adversarial_loss + domain_loss
        return {"total": total_loss, "real": adversarial_real, "fake": adversarial_fake, "domain": domain_loss}

    def get_cycled_images_input(self, bw_source_images, fw_target_domain, input_dropout_mask, fw_genned_image,
                                bw_target_palette, batch_shape):
        """
        Returns a list of tensors that represent the input for the generator to create the cycle images
        :param bw_source_images: batch of source images for the backwards pass (shape=[b, d, s, s, c]). They may have
            a different palette than the fw_source_images if the palette perturbation is enabled
        :param fw_target_domain: batch of target domain indices (shape=[b])
        :param input_dropout_mask: mask for which domain images have been dropped out (due to input dropout and being
            the target domain) (shape=[b, d], with 0s for dropped out images)
        :param fw_genned_image: batch of generated images (shape=[b, s, s, c])
        :param bw_target_palette: batch of target palettes (shape=[b, (nc), c]) as a ragged tensor
        :param batch_shape: tuple representing the shape of the batch
        :return: a list of tensors that can be used as input to the generator so cycle images get created. They contain:
            1. the backwards source images, a batch with size [b*d_target] of images [d, s, s, c]
            2. the backwards target domain (shape=[b*d_target]). It is a range(0, d) repeated for each image in the batch
            3. the backwards target palette, which is the original palette of the source batches in the forward pass
                (i.e. fw_source_palette) (shape=[b*d_target, (nc), c]). It is repeated to match the batch size
        """
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]
        bw_target_domain = tf.range(number_of_domains)
        bw_target_domain = tf.tile(bw_target_domain[tf.newaxis, ...], [batch_size, 1])
        # bw_target_domain (shape=[b, d]) with an index per image and domain in the batch

        eliminate_fw_target_mask = tf.not_equal(bw_target_domain, fw_target_domain[..., tf.newaxis])
        bw_target_domain = tf.reshape(tf.boolean_mask(bw_target_domain, eliminate_fw_target_mask),
                                      [batch_size, number_of_domains - 1])
        # bw_target_domain (shape=[b, d_target=d-1]) -- eliminates the redundant (and incorrect) bw_target == fw_target

        bw_target_domain_mask = tf.one_hot(bw_target_domain, number_of_domains, on_value=0., off_value=1.)
        bw_target_domain_mask = bw_target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # bw_target_domain_mask (shape=[b, d_target, d, 1, 1, 1]) with 0s for images that must be suppressed
        # (as they are the target)

        # a. repeat the source images once for each domain, so we can later have an input set with a
        # zeroed backward target for each domain
        repeated_bw_source_images = tf.tile(bw_source_images[:, tf.newaxis, ...],
                                            [1, number_of_domains - 1, 1, 1, 1, 1])
        # repeated_bw_source_images (shape=[b, d_target, d, s, s, c]

        # b. replace the original forward target image with the generated fake image, plus the ones that have
        # been dropped out. it can be only the forward target (--cycled-source-replacer == "forward") or all the
        # source images that have been dropped out due to input dropout (--cycled-source-replacer == "dropout")
        # the Colla's paper does not specify what it does, but the source code¹ uses the "dropout" option
        # ¹ https://github.com/jongcye/CollaGAN_CVPR/blob/master/model/CollaGAN_fExp8.py#L99
        cycled_source_replacement_mask = self.cycled_source_replacer.replace(fw_target_domain, input_dropout_mask)
        fw_genned_image = fw_genned_image[:, tf.newaxis, tf.newaxis, ...]
        # fw_genned_image becomes shape=[b, 1, 1, s, s, c], so it can be broadcast together with repeated_bw_source_images

        bw_source_images = tf.where(cycled_source_replacement_mask, fw_genned_image, repeated_bw_source_images)
        # bw_source_images (shape=[b, d_target, d, s, s, c])

        # c. zero out the images that are the backwards cyclical target
        bw_source_images = bw_source_images * bw_target_domain_mask

        # d. merge the batch and target-domain dimensions to have 4D tensors as expected by the generator
        bw_source_images = tf.reshape(bw_source_images, [
            batch_size * (number_of_domains - 1), number_of_domains, image_size, image_size, channels
        ])
        bw_target_domain = tf.reshape(bw_target_domain, [batch_size * (number_of_domains - 1)])

        bw_target_palette = tf.tile(bw_target_palette, [number_of_domains - 1, 1, 1])
        # bw_target_palette (shape=[b*d_target, (nc), c])

        return self.gen_supplier(bw_source_images, bw_target_domain,
                                 bw_target_palette)

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]

        temperature = self.annealing_scheduler.update(t)

        # 1. select a random target domain with a subset of the images as input
        fw_source_images, fw_target_domain, input_dropout_mask = self.sampler.sample(batch, t)
        # fw_source_images (shape=[b, d, s, s, c])
        # fw_target_domain (shape=[b,])
        # input_dropout_mask (shape=[b, d]), with 0s for images that should be dropped out
        fw_source_palette = palette_utils.batch_extract_palette_ragged(fw_source_images)
        # fw_source_palette (shape=[b, (n), c]) as a ragged tensor

        # 1.5. perturb the palette a bit if inside the probability:
        palette_perturbation_prob = self.config.perturb_palette
        should_perturb_palette = tf.random.uniform(shape=()) < palette_perturbation_prob
        if should_perturb_palette:
            source_images_palette_perturbed, perturbed_palette = palette_utils.batch_perturb_palette(fw_source_images)
            bw_source_images = source_images_palette_perturbed
            fw_target_palette = perturbed_palette
        else:
            bw_source_images = fw_source_images
            fw_target_palette = fw_source_palette

        fw_target_image = tf.gather(bw_source_images, fw_target_domain, batch_dims=1)
        bw_target_palette = fw_source_palette
        # fw_target_image (shape=[b, s, s, c]) is the target for each example in the batch
        # bw_target_images (shape=[b, d, s, s, c]) are the original source images that need to be reconstructed
        # in the backwards pass
        # bw_target_palette (shape=[b, (n), c]) is the original palette used as target palette for the backwards pass

        # fw_source_images_dropped contains, for each domain, an image or zeros with same shape (dropped out)
        fw_source_images_dropped = fw_source_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # fw_source_images_dropped (shape=[b, d, s, s, c], but with all 0s for dropped images)

        with tf.GradientTape(persistent=True) as tape:
            # FORWARD PASS:
            # 1. generate a batch of fake images
            fw_generator_input = self.gen_supplier(fw_source_images_dropped, fw_target_domain, fw_target_palette)
            # fw_generator_input (shape=[b, d, s, s, c])
            fw_genned_image = self.generator(fw_generator_input, training=True)
            # fw_genned_image (shape=[b, s, s, c])

            # BACKWARD PASS:
            # 2. generate a batch of cycled images (back to their source domain)
            bw_generator_input = self.get_cycled_images_input(bw_source_images, fw_target_domain, input_dropout_mask,
                                                              fw_genned_image, bw_target_palette, batch_shape)
            # bw_generator_input (list of [shape=[b*d_target, d, s, s, c], shape=[b*d]])
            bw_genned_images = self.generator(bw_generator_input, training=True)
            # bw_genned_images (shape=[b*d_target, s, s, c])

            # 3. discriminate the real (target) and fake images, then the cycled ones and the source (to train the disc)
            real_predicted_patches, real_predicted_domain = self.discriminator(fw_target_image, training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(fw_genned_image, training=True)
            # xxxx_predicted_patches (shape=[b, 1, 1, 1])
            # xxxx_predicted_domain  (shape=[b, d] -> logits)

            cycled_predicted_patches, cycled_predicted_domain = self.discriminator(bw_genned_images, training=True)
            # cycled_predicted_patches (shape=[b*d_target, 1, 1, 1])
            # cycled_predicted_domain  (shape=[b*d_target, d] -> logits)

            source_predicted_patches, source_predicted_domain = self.discriminator(
                tf.reshape(fw_source_images, [-1, image_size, image_size, channels]),
                training=True)
            # source_predicted_patches (shape=[b*d, 1, 1, 1])
            # source_predicted_domain  (shape=[b*d, d] -> logits)

            # 4. calculate loss terms for the generator
            g_loss = self.generator_loss(fake_predicted_patches, cycled_predicted_patches, fw_genned_image,
                                         fw_target_image, bw_genned_images, fw_source_images, fake_predicted_domain,
                                         cycled_predicted_domain,
                                         fw_target_domain, input_dropout_mask, batch_shape,
                                         fw_target_palette, bw_target_palette, temperature)

            # 5. calculate loss terms for the discriminator
            d_loss = self.discriminator_loss(source_predicted_patches, cycled_predicted_patches,
                                             source_predicted_domain, real_predicted_patches, fake_predicted_patches,
                                             batch_shape)

        generator_gradients = tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        discriminator_gradients = tape.gradient(d_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        summary_step = step // evaluate_steps
        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", g_loss["total"], step=summary_step)
                tf.summary.scalar("adversarial_loss", g_loss["adversarial"], step=summary_step)
                tf.summary.scalar("domain_loss", g_loss["domain"], step=summary_step)
                tf.summary.scalar("ssim_loss", g_loss["ssim"], step=summary_step)
                tf.summary.scalar("l1_forward_loss", g_loss["l1_forward"], step=summary_step)
                tf.summary.scalar("l1_backward_loss", g_loss["l1_backward"], step=summary_step)
                tf.summary.scalar("palette_loss", g_loss["palette"], step=summary_step)
                tf.summary.scalar("histogram_loss", g_loss["histogram"], step=summary_step)
                tf.summary.scalar("weight_decay", g_loss["weight_decay"], step=summary_step)

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", d_loss["total"], step=summary_step)
                tf.summary.scalar("real_loss", d_loss["real"], step=summary_step)
                tf.summary.scalar("fake_loss", d_loss["fake"], step=summary_step)
                tf.summary.scalar("domain_loss", d_loss["domain"], step=summary_step)

    def select_examples_for_visualization(self, train_ds, test_ds):
        number_of_domains = self.config.number_of_domains
        number_of_examples = 3

        ensure_inside_range = lambda x: x % number_of_domains
        train_examples = []
        test_examples = []

        train_ds_iter = train_ds.unbatch().take(number_of_examples).as_numpy_iterator()
        test_ds_iter = test_ds.shuffle(self.config.test_size).unbatch().take(number_of_examples).as_numpy_iterator()
        for c in range(number_of_examples):
            target_index = ensure_inside_range(c + 1)
            train_batch = next(train_ds_iter)
            train_example = (train_batch, target_index)

            test_batch = next(test_ds_iter)
            test_example = (test_batch, target_index)

            train_examples.append(train_example)
            test_examples.append(test_example)

        return train_examples + test_examples

    def preview_generated_images_during_training(self, examples, save_name, step):
        number_of_domains = self.config.number_of_domains
        title = [f"Input {i}" for i in range(number_of_domains)] + ["Target", "Generated"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            if step == 1:
                step = 0
            title[-1] += f" ({step / 1000}k)"
            title[-2] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        for i, example in enumerate(examples):
            domain_images, target_domain = example
            image_size, channels = domain_images[0].shape[0], domain_images[0].shape[2]
            concatenated_images_to_extract_palette = tf.reshape(domain_images,
                                                                (number_of_domains * image_size, image_size, channels))
            concatenated_images_to_extract_palette = dataset_utils.denormalize(concatenated_images_to_extract_palette)
            palette = palette_utils.extract_palette(concatenated_images_to_extract_palette)
            palette = dataset_utils.normalize(tf.cast(palette, tf.float32))

            real_image = domain_images[target_domain]
            domain_images = tf.constant(domain_images)

            # this zeroes out the target image:
            target_domain_mask = tf.one_hot(target_domain, number_of_domains, on_value=0., off_value=1.)
            domain_images *= target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]

            fake_image = self.generator(
                self.gen_supplier(
                    tf.expand_dims(domain_images, 0),
                    tf.expand_dims(target_domain, 0),
                    tf.expand_dims(palette, 0)
                ),
                training=False)

            images = [*domain_images, real_image, fake_image[0]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                if i == 0:
                    if j < number_of_domains:
                        plt.title("Input" if j != target_domain else "Target", fontdict={"fontsize": 24})
                    else:
                        plt.title(title[j], fontdict={"fontsize": 24})
                elif j == target_domain:
                    plt.title("Target", fontdict={"fontsize": 24})

                plt.imshow(tf.clip_by_value(images[j] * 0.5 + 0.5, 0, 1))
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        return figure

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        number_of_domains = self.config.number_of_domains

        def initialize_random_examples_from_dataset(dataset):
            batch = next(iter(dataset.unbatch().batch(num_images).take(1)))
            domain_images = tf.transpose(batch, [1, 0, 2, 3, 4])

            target_domain = tf.random.uniform(shape=[num_images], minval=0, maxval=number_of_domains,
                                              dtype="int32")

            return domain_images, target_domain

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        generator = self.generator

        def generate_images_from_dataset(dataset_name):
            number_of_domains = self.config.number_of_domains
            domain_images, target_domain = example_indices_for_evaluation[dataset_name]
            target_image = tf.gather(domain_images, target_domain, batch_dims=1)

            target_domain_mask = tf.one_hot(target_domain, number_of_domains, on_value=0., off_value=1.)
            domain_images *= target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]

            # call the generator but with batched domain_images and target_domain
            batch_size = 4
            num_batches = tf.shape(domain_images)[0] // batch_size

            fake_images = []
            for i in range(num_batches):
                batch_domain_images = domain_images[i * batch_size:(i + 1) * batch_size]
                batch_target_domain = target_domain[i * batch_size:(i + 1) * batch_size]
                batch_palette = palette_utils.batch_extract_palette_ragged(batch_domain_images)
                fake_image_batch = generator(self.gen_supplier(batch_domain_images, batch_target_domain, batch_palette),
                                             training=False)
                fake_images.append(fake_image_batch)

            fake_images = tf.concat(fake_images, axis=0)
            return target_image, fake_images

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        # for each image i in the dataset...
        for i, domain_images in tqdm(enumerated_dataset, total=num_images):
            # domain_images is a tuple of d images, each with shape [s, s, c]
            one_image_shape = tf.shape(domain_images[0])
            image_size, channels = one_image_shape[0], one_image_shape[-1]
            palette = palette_utils.extract_palette(
                dataset_utils.denormalize(tf.reshape(domain_images, (-1, image_size, channels))))
            palette = dataset_utils.normalize(tf.cast(palette, tf.float32))
            # for each number m of missing domains [1 to d[
            for m in range(1, number_of_domains):
                image_path = os.sep.join([base_image_path, f"{i:04d}_at_step_{step}_missing_{m}.png"])
                fig = plt.figure(figsize=(4 * number_of_domains, 4 * number_of_domains))
                plt.suptitle(f"Missing {m} image(s)", fontdict={"fontsize": 20})
                for target_index in range(number_of_domains):
                    input_dropout_mask = tf.one_hot(target_index, number_of_domains, on_value=0., off_value=1.)

                    # define which domains will be available as sources
                    shuffled_domain_indices = tf.random.shuffle(tf.range(number_of_domains)).numpy().tolist()
                    selected_to_drop = {target_index}
                    while len(selected_to_drop) < m:
                        index_to_drop = shuffled_domain_indices.pop(0)
                        input_dropout_mask *= tf.one_hot(index_to_drop, number_of_domains, on_value=0., off_value=1.)
                        selected_to_drop.add(index_to_drop)

                    # generate the image using the available sources
                    dropped_input_image = domain_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
                    dropped_input_image = tf.expand_dims(dropped_input_image, 0)
                    target_domain = tf.expand_dims(target_index, 0)
                    fake_image = self.generator(
                        self.gen_supplier(dropped_input_image, target_domain, palette[tf.newaxis, ...]), training=False)

                    for source_index in range(number_of_domains):
                        idx = (target_index * number_of_domains) + source_index + 1
                        plt.subplot(number_of_domains, number_of_domains, idx)
                        if target_index == source_index:
                            plt.title("Generated", fontdict={"fontsize": 20})
                            image = tf.squeeze(fake_image)
                        else:
                            if source_index in selected_to_drop:
                                plt.title("Dropped", fontdict={"fontsize": 20})
                            image = domain_images[source_index]

                        plt.imshow(tf.clip_by_value(image * 0.5 + 0.5, 0, 1))
                        plt.axis("off")

                plt.savefig(image_path, transparent=True)
                plt.close(fig)

        print(f"Generated {(i + 1) * number_of_domains * (number_of_domains - 1)} images in the test-images folder.")

    def debug_discriminator_output(self, batch, image_path):
        # batch (shape=(b, d, s, s, c))
        batch_transpose = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch_transpose (shape=(d, b, s, s, c))
        batch_shape = tf.shape(batch_transpose)
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]

        palettes = palette_utils.batch_extract_palette_ragged(
            tf.reshape(tf.constant(batch), [batch_size, -1, image_size, channels]))
        domain_images, target_domain, _ = self.sampler.sample(batch_transpose, 0.5)
        # domain_images (shape=[b, d, s, s, c])
        # target_domain (shape=[b,])

        source_domain = tf.math.floormod(target_domain + 1, number_of_domains)
        forward__target_mask = tf.one_hot(target_domain, number_of_domains, on_value=0., off_value=1.)
        backward_target_mask = tf.one_hot(source_domain, number_of_domains, on_value=0., off_value=1.)
        # forward__target_mask (shape=[b, d])
        # backward_target_mask (shape=[b, d])

        forward__input_image = domain_images * forward__target_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # forward__input_image (shape=[b, d, s, s, c], but with all 0s for the target image)

        # real_image is the target for each example in the batch
        real_image = tf.gather(domain_images, target_domain, batch_dims=1)
        fake_image = self.generator(self.gen_supplier(forward__input_image, target_domain, palettes), training=False)

        forward__target_mask = tf.cast(1. - forward__target_mask, tf.bool)
        backward_input_image = domain_images * backward_target_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        backward_input_image = tf.where(
            forward__target_mask[..., tf.newaxis, tf.newaxis, tf.newaxis],
            fake_image[:, tf.newaxis, ...],
            backward_input_image)

        back_image = self.generator(self.gen_supplier(backward_input_image, source_domain, palettes), training=False)

        real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
        fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)
        back_predicted_patches, back_predicted_domain = self.discriminator(back_image, training=True)

        real_predicted_mean = tf.reduce_mean(real_predicted_patches, axis=[1, 2, 3])
        fake_predicted_mean = tf.reduce_mean(fake_predicted_patches, axis=[1, 2, 3])
        back_predicted_mean = tf.reduce_mean(back_predicted_patches, axis=[1, 2, 3])

        real_predicted_domain_logit = tf.math.reduce_max(tf.nn.softmax(real_predicted_domain), axis=1)
        fake_predicted_domain_logit = tf.math.reduce_max(tf.nn.softmax(fake_predicted_domain), axis=1)
        back_predicted_domain_logit = tf.math.reduce_max(tf.nn.softmax(back_predicted_domain), axis=1)

        real_predicted_domain = tf.math.argmax(real_predicted_domain, axis=1, output_type=tf.int32)
        fake_predicted_domain = tf.math.argmax(fake_predicted_domain, axis=1, output_type=tf.int32)
        back_predicted_domain = tf.math.argmax(back_predicted_domain, axis=1, output_type=tf.int32)

        # lsgan yields an unbounded real number, which should be 1 for real images and 0 for fake
        # but, we need to provide them in the [0, 1] range
        concatenated_predictions = tf.concat([real_predicted_patches, fake_predicted_patches, back_predicted_patches],
                                             axis=-1)
        min_value = tf.math.reduce_min(concatenated_predictions)
        max_value = tf.math.reduce_max(concatenated_predictions)
        amplitude = max_value - min_value
        real_predicted = (real_predicted_patches - min_value) / amplitude
        fake_predicted = (fake_predicted_patches - min_value) / amplitude
        back_predicted = (back_predicted_patches - min_value) / amplitude

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted)[-2]
        lower_bound_scaling_factor = image_size // num_patches
        pad_before = (image_size - num_patches * lower_bound_scaling_factor) // 2
        pad_after_ = (image_size - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted = tf.repeat(tf.repeat(real_predicted, lower_bound_scaling_factor, axis=1),
                                   lower_bound_scaling_factor, axis=2)
        real_predicted = tf.pad(real_predicted, [[0, 0], [pad_before, pad_after_], [pad_before, pad_after_], [0, 0]])
        fake_predicted = tf.repeat(tf.repeat(fake_predicted, lower_bound_scaling_factor, axis=1),
                                   lower_bound_scaling_factor, axis=2)
        fake_predicted = tf.pad(fake_predicted, [[0, 0], [pad_before, pad_after_], [pad_before, pad_after_], [0, 0]])
        back_predicted = tf.repeat(tf.repeat(back_predicted, lower_bound_scaling_factor, axis=1),
                                   lower_bound_scaling_factor, axis=2)
        back_predicted = tf.pad(back_predicted, [[0, 0], [pad_before, pad_after_], [pad_before, pad_after_], [0, 0]])

        # display the images: source / real / discr. real / fake / discr. fake
        titles = ["Target", "Disc. Target", "Generated", "Disc. Gen.", "Cycled", "Disc. Cyc."]
        num_rows = batch_size.numpy()
        num_cols = len(titles)
        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))

        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows, num_cols, (i * num_cols) + j + 1)
                subplot_title = ""
                if i == 0:
                    subplot_title = titles[j]
                if titles[j] == "Disc. Target":
                    subplot_title = tf.strings.join([
                        titles[j],
                        " ",
                        tf.strings.as_string(real_predicted_mean[i], precision=3),
                        " (", self.config.domains[real_predicted_domain[i]],
                        " ",
                        tf.strings.as_string(real_predicted_domain_logit[i], precision=2),
                        ")"]).numpy().decode("utf-8")

                elif titles[j] == "Disc. Gen.":
                    subplot_title = tf.strings.join([
                        titles[j],
                        " ",
                        tf.strings.as_string(fake_predicted_mean[i], precision=3),
                        " (", self.config.domains[fake_predicted_domain[i]],
                        " ",
                        tf.strings.as_string(fake_predicted_domain_logit[i], precision=2),
                        ")"]).numpy().decode("utf-8")

                elif titles[j] == "Cycled":
                    current_source_domain_index = tf.cast(source_domain[i], tf.int32)
                    subplot_title = tf.strings.join([
                        titles[j],
                        " (", self.config.domains[current_source_domain_index], ")"]).numpy().decode("utf-8")
                elif titles[j] == "Disc. Cyc.":
                    subplot_title = tf.strings.join([
                        titles[j],
                        " ",
                        tf.strings.as_string(back_predicted_mean[i], precision=3),
                        " (", self.config.domains[back_predicted_domain[i]],
                        " ",
                        tf.strings.as_string(back_predicted_domain_logit[i], precision=2),
                        ")"]).numpy().decode("utf-8")

                plt.title(subplot_title, fontdict={"fontsize": 20})
                image = None
                imshow_args = {}
                if titles[j] == "Target":
                    image = real_image[i] * 0.5 + 0.5
                elif titles[j] == "Disc. Target":
                    image = real_predicted[i]
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                elif titles[j] == "Generated":
                    image = fake_image[i] * 0.5 + 0.5
                elif titles[j] == "Disc. Gen.":
                    image = fake_predicted[i]
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                elif titles[j] == "Cycled":
                    image = back_image[i] * 0.5 + 0.5
                elif titles[j] == "Disc. Cyc.":
                    image = back_predicted[i]
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

                plt.imshow(image, **imshow_args)
                plt.axis("off")

        plt.savefig(image_path, transparent=True)
        plt.close(fig)


class CollaGANModelShuffledBatches(CollaGANModel):
    def __init__(self, config):
        super(CollaGANModelShuffledBatches, self).__init__(config)

    def generator_loss(self, fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                       cycled_images, source_images_5d,
                       fake_predicted_domain, cycle_predicted_domain, target_domain,
                       input_dropout_mask, batch_shape, fw_target_palette, bw_target_palette, temperature):
        # cycled_images (shape=[b*d, s, s, c])
        # input_dropout_mask (shape=[b, d])
        number_of_domains = batch_shape[0]
        batch_size, image_size, channels = batch_shape[1], batch_shape[2], batch_shape[4]
        number_of_domains_float = tf.cast(number_of_domains, tf.float32)

        # the source_images need to have the forward target images excluded so it can be properly compared
        # against the cycled_images
        bw_target_domain = tf.tile(tf.range(number_of_domains)[tf.newaxis, ...], [batch_size, 1])
        eliminate_fw_target_mask = tf.not_equal(bw_target_domain, target_domain[..., tf.newaxis])
        bw_target_domain = tf.reshape(tf.boolean_mask(bw_target_domain, eliminate_fw_target_mask),
                                      [batch_size, number_of_domains - 1])

        source_images_5d = tf.gather(source_images_5d, bw_target_domain, batch_dims=1)
        source_images = tf.reshape(source_images_5d,
                                   [batch_size * (number_of_domains - 1), image_size, image_size, channels])
        cycled_images_5d = tf.reshape(cycled_images,
                                      [batch_size, number_of_domains - 1, image_size, image_size, channels])

        input_dropout_mask = tf.gather(input_dropout_mask, bw_target_domain, batch_dims=1)
        input_dropout_mask_1d = tf.reshape(input_dropout_mask, [batch_size * (number_of_domains - 1)])
        # source_images (shape=[b, d, s, s, c])
        # cycle_images_5d (shape=[b, d, s, s, c])
        # input_dropout_mask_1d (shape=[b*d])

        # adversarial (lsgan) loss
        adversarial_forward__loss = tf.reduce_mean(tf.math.squared_difference(fake_predicted_patches, 1.))
        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(cycled_predicted_patches, 1.)) * \
                                    (number_of_domains_float - 1.)
        adversarial_loss = (adversarial_forward__loss + adversarial_backward_loss) / number_of_domains_float

        # l1 (forward, backward)
        l1_forward__loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        # l1_backward_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         # mean of pixel l1s per image, but 0 for dropped out input images
        #         tf.reduce_mean(tf.abs(source_images_5d - cycled_images_5d), axis=[2, 3, 4]) * input_dropout_mask,
        #         axis=1) * tf.reduce_sum(input_dropout_mask, axis=1),
        #     axis=0)
        source_images_5d_kept = tf.boolean_mask(source_images_5d, tf.cast(input_dropout_mask, tf.int32))
        cycled_images_5d_kept = tf.boolean_mask(cycled_images_5d, tf.cast(input_dropout_mask, tf.int32))
        l1_backward_loss = tf.reduce_mean(tf.abs(source_images_5d_kept - cycled_images_5d_kept)) * \
                           tf.reduce_sum(input_dropout_mask)

        # ssim loss (forward, backward)
        ssim_forward_ = tf.image.ssim(fake_image + 1., real_image + 1., 2)
        ssim_backward = tf.image.ssim(cycled_images + 1., source_images + 1., 2) * input_dropout_mask_1d
        # ssim_forward_ (shape=[b,])
        # ssim_backward (shape=[b*d,])
        ssim_forward__loss = tf.reduce_mean(-tf.math.log((1. + ssim_forward_) / 2.))
        # ssim_forward_loss (shape=[b,])
        ssim_backward_loss = tf.reduce_mean(-tf.math.log((1. + ssim_backward) / 2.))
        # ssim_backward_loss (shape=[b,])
        ssim_loss = (ssim_forward__loss + ssim_backward_loss * (number_of_domains_float - 1)) / number_of_domains_float

        # domain classification loss (forward, backward)
        forward__domain = tf.one_hot(target_domain, number_of_domains)
        backward_domain = tf.reshape(tf.one_hot(bw_target_domain, number_of_domains),
                                     [batch_size * (number_of_domains - 1), number_of_domains])
        backward_predicted_domain = cycle_predicted_domain
        # forward__domain (shape=[b, d])
        # backward_domain (shape=[b*d, d])

        classification_forward__loss = self.cce(forward__domain, fake_predicted_domain)
        classification_backward_loss = self.cce(backward_domain, backward_predicted_domain) * \
                                       (number_of_domains_float - 1.)
        classification_loss = (classification_forward__loss + classification_backward_loss) / number_of_domains_float

        # palette loss (forward, backward)
        palette_forward_ = palette_utils.calculate_palette_coverage_loss_ragged(fake_image, fw_target_palette,
                                                                                temperature)
        palette_backward = palette_utils.calculate_palette_coverage_loss_ragged(cycled_images,
                                                                                tf.tile(bw_target_palette,
                                                                                        [number_of_domains - 1, 1, 1]),
                                                                                temperature)
        # palette_forward_ = 0.
        # palette_backward = 0.
        palette_loss = palette_forward_ + palette_backward

        # histogram loss (forward)
        real_histogram = histogram_utils.calculate_rgbuv_histogram(real_image)
        fake_histogram = histogram_utils.calculate_rgbuv_histogram(fake_image)
        histogram_loss = histogram_utils.hellinger_loss(real_histogram, fake_histogram)
        # histogram_loss = 0.

        # regularization loss (l2 - weight decay)
        regularization_loss = tf.reduce_sum(self.generator.losses)

        # observation: ssim loss uses only the backward (cycled) images... that's on the colla's code and paper
        total_loss = self.lambda_adversarial * adversarial_loss + \
                     self.lambda_l1 * l1_forward__loss + self.lambda_l1_backward * l1_backward_loss + \
                     self.lambda_ssim * ssim_backward_loss + \
                     self.lambda_domain * classification_loss + \
                     self.lambda_palette * palette_loss + \
                     self.lambda_histogram * histogram_loss + \
                     self.lambda_regularization * regularization_loss

        return {"total": total_loss, "adversarial": adversarial_loss, "l1_forward": l1_forward__loss,
                "l1_backward": l1_backward_loss, "ssim": ssim_loss, "domain": classification_loss,
                "palette": palette_loss, "histogram": histogram_loss, "weight_decay": regularization_loss}

    def discriminator_loss(self, source_predicted_patches, cycled_predicted_patches, source_predicted_domain,
                           source_label_domain, real_predicted_patches, fake_predicted_patches, batch_shape):
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        adversarial_real = tf.reduce_mean(tf.math.squared_difference(real_predicted_patches, 1.))
        adversarial_fake = tf.reduce_mean(tf.math.square(fake_predicted_patches))
        adversarial_forward_loss = adversarial_real + adversarial_fake

        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(source_predicted_patches, 1.)) + \
                                    tf.reduce_mean(tf.math.square(cycled_predicted_patches))
        adversarial_backward_loss *= tf.cast(number_of_domains, tf.float32)

        adversarial_loss = (adversarial_forward_loss + adversarial_backward_loss) / \
                           (tf.cast(number_of_domains, tf.float32) + 1.)

        source_label_domain = tf.one_hot(source_label_domain, number_of_domains)
        # source_label_domain (shape=[b*d, d])
        domain_loss = self.cce(source_label_domain, source_predicted_domain)

        total_loss = adversarial_loss + domain_loss
        return {"total": total_loss, "real": adversarial_real, "fake": adversarial_fake, "domain": domain_loss}

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]

        temperature = self.annealing_scheduler.update(t)

        # 1. select a random target domain with a subset of the images as input
        fw_source_images, fw_target_domain, input_dropout_mask = self.sampler.sample(batch, t)
        # fw_source_images (shape=[b, d, s, s, c])
        # fw_target_domain (shape=[b,])
        # input_dropout_mask (shape=[b, d]), with 0s for images that should be dropped out
        fw_source_palette = palette_utils.batch_extract_palette_ragged(fw_source_images)
        # fw_source_palette (shape=[b, (n), c]) as a ragged tensor

        # 1.5. perturb the palette a bit if inside the probability:
        palette_perturbation_prob = self.config.perturb_palette
        should_perturb_palette = tf.random.uniform(shape=()) < palette_perturbation_prob
        if should_perturb_palette:
            source_images_palette_perturbed, perturbed_palette = palette_utils.batch_perturb_palette(fw_source_images)
            bw_source_images = source_images_palette_perturbed
            fw_target_palette = perturbed_palette
        else:
            bw_source_images = fw_source_images
            fw_target_palette = fw_source_palette

        fw_target_image = tf.gather(bw_source_images, fw_target_domain, batch_dims=1)
        bw_target_palette = fw_source_palette
        # fw_target_image (shape=[b, s, s, c]) is the target for each example in the batch
        # bw_target_images (shape=[b, d, s, s, c]) are the original source images that need to be reconstructed
        # in the backwards pass
        # bw_target_palette (shape=[b, (n), c]) is the original palette used as target palette for the backwards pass

        # fw_source_images_dropped contains, for each domain, an image or zeros with same shape (dropped out)
        fw_source_images_dropped = fw_source_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # fw_source_images_dropped (shape=[b, d, s, s, c], but with all 0s for dropped images)

        with tf.GradientTape(persistent=True) as tape:
            # FORWARD PASS:
            # 1. generate a batch of fake images
            fw_generator_input = self.gen_supplier(fw_source_images_dropped, fw_target_domain, fw_target_palette)
            # fw_generator_input (shape=[b, d, s, s, c])
            fw_genned_image = self.generator(fw_generator_input, training=True)
            # fw_genned_image (shape=[b, s, s, c])

            # BACKWARD PASS:
            # 2. generate a batch of cycled images (back to their source domain)
            bw_generator_input = self.get_cycled_images_input(bw_source_images, fw_target_domain, input_dropout_mask,
                                                              fw_genned_image, bw_target_palette, batch_shape)
            # bw_generator_input (list of [shape=[b*d_target, d, s, s, c], shape=[b*d]])
            bw_genned_images = self.generator(bw_generator_input, training=True)
            # bw_genned_images (shape=[b*d_target, s, s, c])

            # 3. discriminate the real (target) and fake images, then the cycled ones and the source (to train the disc)
            real_predicted, fake_predicted = self.mix_and_discriminate(fw_target_image, fw_genned_image, batch_size)
            real_predicted_patches, real_predicted_domain = real_predicted
            fake_predicted_patches, fake_predicted_domain = fake_predicted
            # xxxx_predicted_patches (shape=[b, 1, 1, 1])
            # xxxx_predicted_domain  (shape=[b, d] -> logits)

            bw_batch_size = batch_size * (number_of_domains - 1)
            # fw_source_images (shape=[b, d, s, s, c]), and we need to make it shuffled and [b*d_target, s, s, c]
            fw_source_images_flattended = tf.reshape(fw_source_images, [-1, image_size, image_size, channels])
            fw_source_images_shuffle_indices = tf.random.shuffle(tf.range(batch_size * number_of_domains))
            # fw_source_images_shuffle_indices contains shuffled indices from [0, b*d[
            fw_source_images_sampled = tf.gather(fw_source_images_flattended, fw_source_images_shuffle_indices)[
                                       :bw_batch_size]
            fw_source_domain_label = tf.gather(tf.tile(tf.range(number_of_domains), [batch_size]),
                                               fw_source_images_shuffle_indices)[:bw_batch_size]
            # if t >= 0.5:
            #     tf.print("fw_source_domain_label", fw_source_domain_label)
            #     io_utils.show_image_matrix(fw_source_images_sampled)

            source_predicted, cycle_predicted = self.mix_and_discriminate(fw_source_images_sampled, bw_genned_images,
                                                                          bw_batch_size)
            cycle_predicted_patches, cycle_predicted_domain = cycle_predicted
            source_predicted_patches, source_predicted_domain = source_predicted
            # source_predicted_patches (shape=[b*d_target, 1, 1, 1])
            # source_predicted_domain  (shape=[b*d_target, d] -> logits)

            # 4. calculate loss terms for the generator
            g_loss = self.generator_loss(fake_predicted_patches, cycle_predicted_patches, fw_genned_image,
                                         fw_target_image, bw_genned_images, fw_source_images, fake_predicted_domain,
                                         cycle_predicted_domain,
                                         fw_target_domain, input_dropout_mask, batch_shape,
                                         fw_target_palette, bw_target_palette, temperature)

            # 5. calculate loss terms for the discriminator
            d_loss = self.discriminator_loss(source_predicted_patches, cycle_predicted_patches,
                                             source_predicted_domain, fw_source_domain_label,
                                             real_predicted_patches, fake_predicted_patches,
                                             batch_shape)

        generator_gradients = tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        discriminator_gradients = tape.gradient(d_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        summary_step = step // evaluate_steps
        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", g_loss["total"], step=summary_step)
                tf.summary.scalar("adversarial_loss", g_loss["adversarial"], step=summary_step)
                tf.summary.scalar("domain_loss", g_loss["domain"], step=summary_step)
                tf.summary.scalar("ssim_loss", g_loss["ssim"], step=summary_step)
                tf.summary.scalar("l1_forward_loss", g_loss["l1_forward"], step=summary_step)
                tf.summary.scalar("l1_backward_loss", g_loss["l1_backward"], step=summary_step)
                tf.summary.scalar("palette_loss", g_loss["palette"], step=summary_step)
                tf.summary.scalar("histogram_loss", g_loss["histogram"], step=summary_step)
                tf.summary.scalar("weight_decay", g_loss["weight_decay"], step=summary_step)

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", d_loss["total"], step=summary_step)
                tf.summary.scalar("real_loss", d_loss["real"], step=summary_step)
                tf.summary.scalar("fake_loss", d_loss["fake"], step=summary_step)
                tf.summary.scalar("domain_loss", d_loss["domain"], step=summary_step)

    def mix_and_discriminate(self, real_image, fake_image, half_batch_size):
        batch_size = half_batch_size * 2
        discriminator_input = tf.concat([real_image, fake_image], axis=0)

        # create shuffled indices and remember the original positions (through tf.argsort(shuffled))
        shuffled_indices = tf.random.shuffle(tf.range(batch_size))
        discriminator_input = tf.gather(discriminator_input, shuffled_indices)
        inverse_indices = tf.argsort(shuffled_indices)

        # discriminate the shuffled combined half batches
        predicted_patches, predicted_domain = self.discriminator(discriminator_input, training=True)

        # Get the predictions in the original order
        patches_ordered = tf.gather(predicted_patches, inverse_indices)
        domain_ordered = tf.gather(predicted_domain, inverse_indices)

        # split in real and fake predictions
        real_predicted_patches = patches_ordered[:half_batch_size]
        real_predicted_domain = domain_ordered[:half_batch_size]
        fake_predicted_patches = patches_ordered[half_batch_size:]
        fake_predicted_domain = domain_ordered[half_batch_size:]

        return (real_predicted_patches, real_predicted_domain), (fake_predicted_patches, fake_predicted_domain)


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch, t):
        pass

    def random_target_index(self, batch_size):
        return tf.random.uniform(shape=[batch_size], maxval=self.config.number_of_domains, dtype=tf.int32)


class InputDropoutSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)
        # a list shape=(d, to_drop, ?, d) that is, per possible target pose index (first dimension),
        #     for each possible number of dropped inputs (second dimension): all permutations of a boolean array that
        #     (a) nullifies the target index and (b) nullifies a number of additional inputs equal to 0, 1 or 2
        #     (determined by inputs_to_drop).
        dropout_null_list = dataset_utils.create_input_dropout_index_list([1, 2, 3], self.config.number_of_domains)
        self.null_list = tf.ragged.constant(dropout_null_list, ragged_rank=2, dtype="bool")

    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        random_values = tf.random.uniform(shape=[batch_size])
        max_values = tf.cast(dropout_null_list_for_target.row_lengths(axis=1), tf.float32)
        return tf.cast(tf.floor(random_values * max_values), tf.int32) + 1

    def sample(self, batch, t):
        """
        Samples a batch of images and a target domain index, plus a mask for input dropout
        :param batch: tensor of images with shape [d, b, s, s, c]
        :param t: the progress of training between [0, 1]
        :return: tensor of images with shape [b, d, s, s, c], tensor of target domain indices with shape [b,],
                    tensor of input dropout masks with shape [b, d]
        """
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        # reorders the batch from [d, b, s, s, c] to [B, D, s, s, c]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])

        # finds a random target side for each example in the batch
        # (shape=[b,]), eg, [0, 1, 3, 3]
        target_domain_index = self.random_target_index(batch_size)

        # applies input dropout as described in the CollaGAN paper and implemented in the code
        #  this is adapted from getBatch_RGB_varInp in CollaGAN
        #  a. randomly choose an input dropout mask such as [True, False, False, True]
        #     it is done by indexing the null_list dimension by dimension until we have
        #     a tensor with a boolean dropout mask per example in the batch (shape=[b, d])
        dropout_null_list_for_target = tf.gather(
            tf.tile(self.null_list[tf.newaxis, ...], [batch_size, 1, 1, 1, 1]),
            target_domain_index, batch_dims=1)
        # dropout_null_list_for_target (shape=[b, to_drop, ?, d])
        random_number_of_inputs_to_drop = self.select_number_of_inputs_to_drop(batch_size, dropout_null_list_for_target,
                                                                               t)
        dropout_null_list_for_target_and_number_of_inputs = tf.gather(dropout_null_list_for_target,
                                                                      random_number_of_inputs_to_drop - 1,
                                                                      batch_dims=1)
        # dropout_null_list_for_target_and_number_of_inputs (shape=[b, ?, d])
        random_permutation_index = tf.random.uniform(shape=[batch_size])
        max_values = tf.cast(dropout_null_list_for_target_and_number_of_inputs.row_lengths(axis=1), tf.float32)
        random_permutation_index = tf.cast(tf.floor(random_permutation_index * max_values), tf.int32)
        input_dropout_mask = tf.gather(dropout_null_list_for_target_and_number_of_inputs,
                                       random_permutation_index,
                                       batch_dims=1)
        # input_dropout_mask (shape=[b, d])
        input_dropout_mask = tf.where(input_dropout_mask, 0., 1.)

        return batch, target_domain_index, input_dropout_mask


class AggressiveInputDropoutSampler(InputDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # 10% of the time, drop 1 inputs
        # 30% of the time, drop 2 inputs
        # 60% of the time, drop 3 inputs
        u = tf.random.uniform(shape=[batch_size])
        return tf.where(u < 0.1, 1, tf.where(u < 0.4, 2, 3))


class BalancedInputDropoutSampler(InputDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # 43% of the time, drop 3 inputs
        # 43% of the time, drop 2 inputs
        # 14% of the time, drop 1 inputs
        u = tf.random.uniform(shape=[batch_size])
        return tf.where(u < 0.43, 3, tf.where(u < 0.86, 2, 1))


class ConservativeInputDropoutSampler(InputDropoutSampler):
    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # 10% of the time, drop 3 inputs
        # 30% of the time, drop 2 inputs
        # 60% of the time, drop 1 inputs
        u = tf.random.uniform(shape=[batch_size])
        return tf.where(u < 0.1, 3, tf.where(u < 0.4, 2, 1))


class CurriculumLearningSampler(InputDropoutSampler):
    def __init__(self, config):
        super().__init__(config)
        self.balanced_sampler = BalancedInputDropoutSampler(config)

    def select_number_of_inputs_to_drop(self, batch_size, dropout_null_list_for_target, t):
        # start with easy (missing 1) samples, then move to harder ones
        # until 17% of the training, drop 1 inputs
        # until 33% of the training, drop 2 inputs
        # until 50% of the training, drop 3 inputs
        # remainder 50% of the training, drop randomly
        n = self.balanced_sampler.select_number_of_inputs_to_drop(batch_size, dropout_null_list_for_target, t) + 1
        return tf.where(t < 0.166667, 1, tf.where(t < 0.33333, 2, tf.where(t < 0.5, 3, n)))


class SimpleSampler(ExampleSampler):
    def sample(self, batch, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        # reorders the batch from [d, b, s, s, c] to [B, D, s, s, c]
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])

        # finds a random target side for each example in the batch
        # (shape=[b,]), eg, [0, 1, 3, 3]
        target_domain_index = self.random_target_index(batch_size)

        # creates a mask (e {0, 1}) representing which inputs will be provided to the generator for each
        # example in the batch
        # (shape=[b, d]), eg, [[0, 1, 1, 1], ..., [1, 0, 1, 1]]
        input_dropout_mask = tf.one_hot(target_domain_index, number_of_domains)
        input_dropout_mask = tf.where(input_dropout_mask == 1., 0., 1.)

        return batch, target_domain_index, input_dropout_mask


class CycledSourceReplacer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def replace(self, forward_target_domain, input_dropout_mask):
        pass


class ForwardOnlyCycledSourceReplacer(CycledSourceReplacer):
    def __init__(self, config):
        super().__init__(config)

    def replace(self, fw_target_domain, input_dropout_mask):
        number_of_domains = self.config.number_of_domains
        fw_target_domain_mask = tf.one_hot(fw_target_domain, number_of_domains,
                                           dtype=tf.bool, on_value=True, off_value=False)
        fw_target_domain_mask = fw_target_domain_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape
        return fw_target_domain_mask


class DroppedOutCycledSourceReplacer(CycledSourceReplacer):
    def __init__(self, config):
        super().__init__(config)

    def replace(self, forward_target_domain, input_dropout_mask):
        inverted_input_dropout_mask = tf.logical_not(tf.cast(input_dropout_mask, tf.bool))
        inverted_input_dropout_mask = inverted_input_dropout_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape

        return inverted_input_dropout_mask

# Sanity checking collagan's palette conditioning:
# python train.py collagan --generator palette-transformer --rm2k --steps 200 --evaluate-steps 100 --vram 4096 --perturb-palette 0.5 --cycled-source-replacer forward --temperature 0.1 --annealing linear
