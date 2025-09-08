import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from utility import palette_utils, io_utils
from utility.keras_utils import NParamsSupplier
from .networks import stargan_resnet_generator, stargan_resnet_discriminator
from .side2side_model import S2SModel


class UnpairedStarGANModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_gp = config.lambda_gp
        self.lambda_domain = config.lambda_domain
        self.lambda_reconstruction = config.lambda_reconstruction
        self.lambda_palette = config.lambda_palette
        self.lambda_tv = config.lambda_tv
        self.discriminator_steps = config.d_steps
        self.domain_classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if config.sampler == "multi-target":
            self.sampler = MultiTargetSampler(config)
        else:
            self.sampler = SingleTargetSampler(config)
        self.gen_supplier = NParamsSupplier(3 if config.source_domain_aware_generator else 2)
        self.crit_supplier = NParamsSupplier(2 if config.conditional_discriminator else 1)
        self.generator = self.inference_networks["generator"]
        self.discriminator = self.training_only_networks["discriminator"]

    def get_annealing_layers(self):
        return [self.generator.quantization] if self.config.palette_quantization else []

    def create_inference_networks(self):
        config = self.config
        if config.generator == "resnet" or config.generator == "":
            return {
                "generator": stargan_resnet_generator(config.image_size, config.output_channels,
                                                      config.number_of_domains,
                                                      config.source_domain_aware_generator, config.capacity)
            }
        else:
            raise ValueError(f"The provided {config.generator} type for generator has not been implemented.")

    def create_training_only_networks(self):
        config = self.config
        if config.discriminator == "resnet" or config.discriminator == "":
            return {
                "discriminator": stargan_resnet_discriminator(config.number_of_domains, config.image_size,
                                                              config.output_channels,
                                                              config.conditional_discriminator)
            }
        else:
            raise ValueError(f"The provided {config.discriminator} type for discriminator has not been implemented.")

    def generator_loss(self, critic_fake_patches, critic_fake_domain, fake_domain, real_image, recreated_image,
                       fake_image, source_image, target_palette, t):
        adversarial_loss = -tf.reduce_mean(critic_fake_patches)
        domain_loss = tf.reduce_mean(self.domain_classification_loss(fake_domain, critic_fake_domain))
        recreation_loss = tf.reduce_mean(tf.abs(source_image - recreated_image))
        # palette_loss = palette.calculate_palette_loss(fake_image, target_palette)
        # tv_loss = tf.reduce_mean(tf.image.total_variation(fake_image))
        palette_loss = 0.
        tv_loss = 0.

        total_loss = (
                adversarial_loss +
                self.lambda_domain * domain_loss +
                self.lambda_reconstruction * recreation_loss +
                (self.lambda_palette * t) * palette_loss +
                (self.lambda_tv * t) * tv_loss)

        return {"total": total_loss, "adversarial": adversarial_loss, "domain": domain_loss,
                "recreation": recreation_loss, "palette": palette_loss, "total_variation": tv_loss}

    def discriminator_loss(self, critic_real_patches, critic_real_domain, real_domain, critic_fake_patches,
                           gradient_penalty):
        real_loss = -tf.reduce_mean(critic_real_patches)
        fake_loss = tf.reduce_mean(critic_fake_patches)
        domain_loss = tf.reduce_mean(self.domain_classification_loss(real_domain, critic_real_domain))

        total_loss = (
                fake_loss + real_loss +
                self.lambda_domain * domain_loss +
                self.lambda_gp * gradient_penalty)
        return {"total": total_loss, "real": real_loss, "fake": fake_loss, "domain": domain_loss,
                "gp": gradient_penalty}

    def select_examples_for_visualization(self, train_ds, test_ds):
        number_of_domains = self.config.number_of_domains
        # number_of_examples = number_of_domains + 1
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
            train_example = (train_domain_images[source_index], source_index,
                             train_domain_images[target_index], target_index)
            test_domain_images = next(test_ds_iter)
            test_example = (test_domain_images[source_index], source_index,
                            test_domain_images[target_index], target_index)

            train_examples.append(train_example)
            test_examples.append(test_example)

        return train_examples + test_examples

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, channels = batch_shape[0], batch_shape[1], batch_shape[4]

        # back, left, front, right for pixel-sides: [d, b, s, s, c]
        domain_images = batch
        # extracts the palette from the combined images of each sample in the batch: (1) prepare and (2) extract
        # (1) prepare the batch, so it is: [b, d*s*s, c]
        combined_images = tf.reshape(tf.transpose(batch, [1, 0, 2, 3, 4]), [batch_size, -1, channels])
        # (2) extract the palette for each image in the batch: [b, (p), c]
        palettes = palette_utils.batch_extract_palette_ragged(combined_images)

        # TRAINING THE DISCRIMINATOR
        # ==========================
        #
        # 1. select a random source domain with a random target
        source_domain, source_image, target_domain, target_image = self.sampler.sample(domain_images)

        # 2. calculate gradient penalty as we're using wgan-gp to train
        gp_epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        with tf.GradientTape() as disc_tape:
            with tf.GradientTape() as gp_tape:
                genned_image = self.generator(
                    self.gen_supplier(source_image, target_domain, source_domain), training=True)
                fake_image_mixed = gp_epsilon * source_image + (1 - gp_epsilon) * genned_image
                fake_mixed_predicted, _ = self.discriminator(
                    self.crit_supplier(fake_image_mixed, source_image), training=True)

            # computing the gradient penalty from the wasserstein-gp GAN
            gp_grads = gp_tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

            # 3. now we actually do a feed forward with real and fake to the critic and check the loss
            # passing real and fake images through the critic
            real_predicted_patches, real_predicted_domain = self.discriminator(
                self.crit_supplier(target_image, source_image), training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(
                self.crit_supplier(genned_image, source_image), training=True)

            c_loss = self.discriminator_loss(real_predicted_patches, real_predicted_domain,
                                             tf.one_hot(source_domain, number_of_domains),
                                             fake_predicted_patches, gradient_penalty)

        # 4. apply the gradients to the critic weights
        discriminator_gradients = disc_tape.gradient(c_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        d_loss_summaries = {
            "total_loss": c_loss["total"],
            "real_loss": c_loss["real"],
            "fake_loss": c_loss["fake"],
            "real_domain_loss": c_loss["domain"],
            "gradient_penalty": c_loss["gp"],
        }

        # TRAINING THE GENERATOR
        # ======================
        #
        # we only train the generator at every x discriminator update steps
        if step % self.discriminator_steps == 0:
            # 1. use previously selected random generator input (random source image/domain + random target domain)
            with tf.GradientTape() as gen_tape:
                # 2. feed forward the generator and critic
                genned_image = self.generator(self.gen_supplier(source_image, target_domain, source_domain),
                                              training=True)
                fake_predicted_patches, fake_predicted_domain = self.discriminator(
                    self.crit_supplier(genned_image, source_image),
                    training=True)

                # 3. reconstruct the image to the original domain
                remade_image = self.generator(self.gen_supplier(genned_image, source_domain, target_domain),
                                              training=True)

                # 4. calculate the loss
                g_loss = self.generator_loss(fake_predicted_patches, fake_predicted_domain,
                                             tf.one_hot(target_domain, number_of_domains),
                                             target_image, remade_image, genned_image, source_image, palettes, t)

            # 5. update the generator weights using the gradients
            generator_gradients = gen_tape.gradient(g_loss["total"], self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            g_loss_summaries = {
                "total_loss": g_loss["total"],
                "adversarial_loss": g_loss["adversarial"],
                "domain_loss": g_loss["domain"],
                "recreation_loss": g_loss["recreation"],
                "l1_loss": tf.constant(0.),
                "palette_loss": g_loss["palette"],
                "total_variation_loss": g_loss["total_variation"],
            }
        else:
            dummy = tf.constant(0.)
            g_loss_summaries = {
                "total_loss": dummy, 
                "adversarial_loss": dummy, 
                "domain_loss": dummy,
                "recreation_loss": dummy, 
                "l1_loss": dummy,
                "palette_loss": dummy, 
                "total_variation_loss": dummy
            }

        return d_loss_summaries, g_loss_summaries

    def preview_generated_images_during_training(self, examples, save_name, step):
        title = ["Input", "Target", "Generated", "Reconstructed"]
        num_images = len(examples)
        num_columns = len(title)

        if step is not None:
            if step == 1:
                step = 0
            title[-1] += f" ({step / 1000}k)"
            title[-2] += f" ({step / 1000}k)"

        figure = plt.figure(figsize=(4 * num_columns, 4 * num_images))

        for i, example in enumerate(examples):
            source_image, source_domain, target_image, target_domain = example
            target_domain_name = self.config.domains[target_domain]

            source_domain = tf.expand_dims(source_domain, 0)
            target_domain = tf.expand_dims(target_domain, 0)
            source_image = tf.expand_dims(source_image, 0)
            genned_image = self.generator(
                self.gen_supplier(source_image, target_domain, source_domain), training=True)
            remade_image = self.generator(
                self.gen_supplier(genned_image, source_domain[tf.newaxis, ...], target_domain), training=True)

            images = [source_image[0], target_image, genned_image[0], remade_image[0]]
            for j in range(num_columns):
                idx = i * num_columns + j + 1
                plt.subplot(num_images, num_columns, idx)
                if i == 0:
                    plt.title(f"{title[j]}\n{target_domain_name}" if j == 1 else title[j] + "\n",
                              fontdict={"fontsize": 24})
                elif j == 1:
                    plt.title(target_domain_name, fontdict={"fontsize": 24})
                plt.imshow(images[j] * 0.5 + 0.5)
                plt.axis("off")

        figure.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, transparent=True)

        image = io_utils.plot_to_image(figure, self.config.inner_channels)

        return image

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
        generator = self.generator

        def generate_images_from_dataset(dataset_name):
            target_images, source_images, target_domains, source_domains = example_indices_for_evaluation[dataset_name]
            number_of_examples = len(source_images)
            fake_images = np.empty((number_of_examples, self.config.image_size, self.config.image_size,
                                    self.config.output_channels), dtype=np.float32)

            batch_size = self.config.batch
            for batch_start in range(0, number_of_examples, batch_size):
                batch_end = batch_start + batch_size
                batch_end = min(batch_end, number_of_examples)

                source_images_slices = source_images[batch_start:batch_end]
                target_domains_slice = target_domains[batch_start:batch_end]
                source_domains_slice = source_domains[batch_start:batch_end]

                fake_images_slice = generator(
                    self.gen_supplier(source_images_slices, target_domains_slice, source_domains_slice), training=True)
                fake_images[batch_start:batch_end] = fake_images_slice

            logging.debug(
                f"Generated all {number_of_examples} fake images from {dataset_name}, which occupy "
                f"{fake_images.nbytes / 1024 / 1024} MB.")
            return target_images, tf.constant(fake_images)

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def debug_discriminator_output(self, batch, image_path):
        # generates the fake image and the discriminations of the real and fake
        domain_images = batch
        number_of_domains = self.config.number_of_domains
        batch_size, image_size = tf.shape(batch)[0], tf.shape(batch)[2]

        source_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=number_of_domains, dtype="int32")
        target_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=number_of_domains, dtype="int32")
        source_images = tf.gather(domain_images, source_indices, batch_dims=1)
        target_images = tf.gather(domain_images, target_indices, batch_dims=1)
        fake_images = self.generator(self.gen_supplier(source_images, target_indices, source_indices), training=True)

        real_predicted_patches, real_predicted_domain = self.discriminator(
            self.crit_supplier(target_images, source_images), training=True)
        fake_predicted_patches, fake_predicted_domain = self.discriminator(
            self.crit_supplier(fake_images, source_images), training=True)

        # finds the mean value of the patches (to display on the titles)
        real_predicted_patches = tf.math.sigmoid(real_predicted_patches)
        fake_predicted_patches = tf.math.sigmoid(fake_predicted_patches)
        real_predicted_mean = tf.reduce_mean(real_predicted_patches, axis=[1, 2, 3])
        fake_predicted_mean = tf.reduce_mean(fake_predicted_patches, axis=[1, 2, 3])

        real_predicted_domain = tf.math.argmax(real_predicted_domain, axis=1, output_type=tf.int32)
        fake_predicted_domain = tf.math.argmax(fake_predicted_domain, axis=1, output_type=tf.int32)

        # makes the patches have the same resolution as the real/fake images by repeating and tiling
        num_patches = tf.shape(real_predicted_patches)[1]
        lower_bound_scaling_factor = image_size // num_patches
        pad_before = (image_size - num_patches * lower_bound_scaling_factor) // 2
        pad_after = (image_size - num_patches * lower_bound_scaling_factor) - pad_before

        real_predicted_patches = tf.repeat(tf.repeat(real_predicted_patches, lower_bound_scaling_factor, axis=1),
                                           lower_bound_scaling_factor, axis=2)
        real_predicted_patches = tf.pad(real_predicted_patches,
                                        [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        real_predicted_patches = real_predicted_patches[..., 0]
        fake_predicted_patches = tf.repeat(tf.repeat(fake_predicted_patches, lower_bound_scaling_factor, axis=1),
                                           lower_bound_scaling_factor, axis=2)
        fake_predicted_patches = tf.pad(fake_predicted_patches,
                                        [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        fake_predicted_patches = fake_predicted_patches[..., 0]

        # display the images: source / real / discr. real / fake / discr. fake
        titles = ["Input", "Target", "Disc. Target", "Generated", "Disc. Gen."]
        num_rows = batch_size.numpy()
        num_cols = len(titles)
        fig = plt.figure(figsize=(4 * 5, 4 * batch_size))

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
                        " (", self.config.domains[real_predicted_domain[i]], ")"]).numpy().decode("utf-8")

                elif titles[j] == "Disc. Gen.":
                    subplot_title = tf.strings.join([
                        titles[j],
                        " ",
                        tf.strings.as_string(fake_predicted_mean[i], precision=3),
                        " (", self.config.domains[fake_predicted_domain[i]], ")"]).numpy().decode("utf-8")

                plt.title(subplot_title, fontdict={"fontsize": 20})
                image = None
                imshow_args = {}
                if titles[j] == "Input":
                    image = source_images[i] * 0.5 + 0.5
                elif titles[j] == "Target":
                    image = target_images[i] * 0.5 + 0.5
                elif titles[j] == "Disc. Target":
                    image = real_predicted_patches[i]
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
                elif titles[j] == "Generated":
                    image = fake_images[i] * 0.5 + 0.5
                elif titles[j] == "Disc. Gen.":
                    image = fake_predicted_patches[i]
                    imshow_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

                plt.imshow(image, **imshow_args)
                plt.axis("off")

        plt.savefig(image_path, transparent=True)
        plt.close(fig)

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        # for each image in the dataset...
        for i, domain_images in tqdm(enumerated_dataset, total=num_images):
            image_path = os.sep.join([base_image_path, f"{i:04d}_at_step_{step}.png"])
            fig = plt.figure(figsize=(4 * number_of_domains, 4 * number_of_domains))
            for source_index in range(number_of_domains):
                source_image = tf.gather(domain_images, source_index)
                for target_index in range(number_of_domains):
                    idx = (source_index * number_of_domains) + target_index + 1
                    plt.subplot(number_of_domains, number_of_domains, idx)
                    plt.title("Input" if source_index == target_index else "")

                    if source_index == target_index:
                        image = source_image
                    else:
                        source_input = tf.expand_dims(source_image, 0)
                        target_domain = tf.expand_dims(target_index, 0)
                        source_domain = tf.expand_dims(source_index, 0)
                        generated_image = self.generator(
                            self.gen_supplier(source_input, target_domain, source_domain), training=True)
                        image = generated_image
                    plt.imshow(tf.squeeze(image) * 0.5 + 0.5)
                    plt.axis("off")
            plt.savefig(image_path, transparent=True)
            plt.close(fig)

        logging.info(f"Generated {(i + 1) * number_of_domains} images in the test-images folder.")


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch):
        pass


class SingleTargetSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)

    def sample(self, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        random_source_index = tf.random.uniform(shape=[], dtype="int32", minval=0, maxval=number_of_domains)
        random_source_index = tf.tile(random_source_index[tf.newaxis, ...], [batch_size, ])
        transposed_batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        random_source_image = tf.gather(transposed_batch, random_source_index, axis=1, batch_dims=1)
        random_target_index = tf.random.uniform(shape=[], dtype="int32", minval=0, maxval=number_of_domains)
        random_target_index = tf.tile(random_target_index[tf.newaxis, ...], [batch_size, ])
        random_target_image = tf.gather(transposed_batch, random_target_index, axis=1, batch_dims=1)

        return random_source_index, random_source_image, random_target_index, random_target_image


class MultiTargetSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)

    def sample(self, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        transposed_batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        random_source_index = tf.random.uniform(shape=[batch_size], dtype="int32", minval=0, maxval=number_of_domains)
        random_source_image = tf.gather(transposed_batch, random_source_index, axis=1, batch_dims=1)
        random_target_index = tf.random.uniform(shape=[batch_size], dtype="int32", minval=0, maxval=number_of_domains)
        random_target_image = tf.gather(transposed_batch, random_target_index, axis=1, batch_dims=1)

        return random_source_index, random_source_image, random_target_index, random_target_image


class PairedStarGANModel(UnpairedStarGANModel):
    def __init__(self, config):
        super().__init__(config)
        self.lambda_l1 = config.lambda_l1

    def generator_loss(self, critic_fake_patches, critic_fake_domain, fake_domain, target_image, recreated_image,
                       fake_image, source_image, target_palette, t):
        g_loss = super().generator_loss(critic_fake_patches, critic_fake_domain, fake_domain, target_image,
                                        recreated_image, fake_image, source_image, target_palette, t)
        l1_loss = tf.reduce_mean(tf.abs(target_image - fake_image))

        total_loss = g_loss["total"] + self.lambda_l1 * l1_loss
        return {"total": total_loss, "adversarial": g_loss["adversarial"], "domain": g_loss["domain"],
                "recreation": g_loss["recreation"], "l1": l1_loss, "palette": g_loss["palette"],
                "total_variation": g_loss["total_variation"]}

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, channels = batch_shape[0], batch_shape[1], batch_shape[4]

        # back, left, front, right for pixel-sides
        domain_images = batch

        # extracts the palette from the combined images of each sample in the batch: (1) prepare and (2) extract
        # (1) prepare the batch, so it is: [b, d*s*s, c]
        combined_images = tf.reshape(tf.transpose(batch, [1, 0, 2, 3, 4]), [batch_size, -1, channels])
        # (2) extract the palette for each image in the batch: [b, (p), c]
        palettes = palette_utils.batch_extract_palette_ragged(combined_images)

        # 1. select a random source domain with a random target
        source_domain, source_image, target_domain, target_image = self.sampler.sample(domain_images)

        # 2. calculate gradient penalty as we're using wgan-gp to train
        gp_epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            with tf.GradientTape() as gp_tape:
                genned_image = self.generator(self.gen_supplier(source_image, target_domain, source_domain),
                                              training=True)
                remade_image = self.generator(self.gen_supplier(genned_image, source_domain, target_domain),
                                              training=True)
                fake_image_mixed = gp_epsilon * source_image + (1 - gp_epsilon) * genned_image
                fake_mixed_predicted, _ = self.discriminator(
                    self.crit_supplier(fake_image_mixed, source_image), training=True)

            # computing the gradient penalty from the wasserstein-gp GAN
            gp_grads = gp_tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

            # 3. now we actually do a feed forward with real and fake to the critic and check the loss
            real_predicted_patches, real_predicted_domain = self.discriminator(
                self.crit_supplier(source_image, target_image), training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(
                self.crit_supplier(genned_image, source_image), training=True)

            c_loss = self.discriminator_loss(real_predicted_patches, real_predicted_domain,
                                             tf.one_hot(source_domain, number_of_domains),
                                             fake_predicted_patches, gradient_penalty)
            g_loss = self.generator_loss(fake_predicted_patches, fake_predicted_domain,
                                         tf.one_hot(target_domain, number_of_domains),
                                         target_image, remade_image, genned_image, source_image, palettes, t)

        # 4. apply the gradients to the critic and generator weights
        critic_gradients = disc_tape.gradient(c_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(critic_gradients, self.discriminator.trainable_variables))

        generator_gradients = gen_tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        c_loss_summaries = {
            "total_loss": c_loss["total"],
            "real_loss": c_loss["real"],
            "fake_loss": c_loss["fake"],
            "real_domain_loss": c_loss["domain"],
            "gradient_penalty": c_loss["gp"]
        }

        g_loss_summaries = {
            "total_loss": g_loss["total"],
            "adversarial_loss": g_loss["adversarial"],
            "domain_loss": g_loss["domain"],
            "recreation_loss": g_loss["recreation"],
            "l1_loss": g_loss["l1"],
            "palette_loss": g_loss["palette"],
            "tv_loss": g_loss["total_variation"]
        }

        return c_loss_summaries, g_loss_summaries
