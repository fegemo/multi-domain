import os
from abc import ABC, abstractmethod

import tensorflow as tf
from matplotlib import pyplot as plt

import io_utils
from networks import stargan_resnet_generator, stargan_resnet_discriminator
from side2side_model import S2SModel


class UnpairedStarGANModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_gp = config.lambda_gp
        self.lambda_domain = config.lambda_domain
        self.lambda_reconstruction = config.lambda_reconstruction
        self.discriminator_steps = config.d_steps
        self.domain_classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if config.sampler == "multi-target":
            self.sampler = MultiTargetSampler(config)
        else:
            self.sampler = SingleTargetSampler(config)

    @property
    def models(self):
        return [self.discriminator, self.generator]

    def create_generator(self):
        config = self.config
        if config.generator == "resnet" or config.generator == "":
            return stargan_resnet_generator(config.image_size, config.output_channels, config.number_of_domains)
        # elif config.generator_type == "unet":
        #     return StarGANUnetGenerator()
        else:
            raise ValueError(f"The provided {config.generator} type for generator has not been implemented.")

    def create_discriminator(self):
        config = self.config
        if config.discriminator == "resnet" or config.discriminator == "":
            return stargan_resnet_discriminator(config.number_of_domains)
        else:
            raise ValueError(f"The provided {config.discriminator} type for discriminator has not been implemented.")

    def generator_loss(self, critic_fake_patches, critic_fake_domain, fake_domain, real_image, reconstructed_image):
        fake_loss = -tf.reduce_mean(critic_fake_patches)
        fake_domain_loss = tf.reduce_mean(self.domain_classification_loss(fake_domain, critic_fake_domain))
        reconstruction_loss = tf.reduce_mean(tf.abs(real_image - reconstructed_image))

        total_loss = fake_loss + \
                     self.lambda_domain * fake_domain_loss + \
                     self.lambda_reconstruction * reconstruction_loss
        return total_loss, fake_loss, fake_domain_loss, reconstruction_loss

    def discriminator_loss(self, critic_real_patches, critic_real_domain, real_domain, critic_fake_patches,
                           gradient_penalty):
        real_loss = -tf.reduce_mean(critic_real_patches)
        fake_loss = tf.reduce_mean(critic_fake_patches)
        real_domain_loss = tf.reduce_mean(self.domain_classification_loss(real_domain, critic_real_domain))

        total_loss = fake_loss + real_loss + \
                     self.lambda_domain * real_domain_loss + \
                     self.lambda_gp * gradient_penalty
        return total_loss, real_loss, fake_loss, real_domain_loss, gradient_penalty

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
                source_index = 2
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
    def train_step(self, batch, step, update_steps):
        batch_size = tf.shape(batch[0])[0]

        # back, left, front, right for pixel-sides
        domain_images = batch

        # TRAINING THE DISCRIMINATOR
        # ==========================
        #
        # 1. select a random source domain with a random target
        random_input, real_domain, real_image, target_domain = self.sampler.sample(domain_images)

        # 2. calculate gradient penalty as we're using wgan-gp to train
        gp_epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        with tf.GradientTape() as disc_tape:
            with tf.GradientTape() as gp_tape:
                fake_image = self.generator(random_input, training=True)
                fake_image_mixed = gp_epsilon * real_image + (1 - gp_epsilon) * fake_image
                fake_mixed_predicted, _ = self.discriminator(fake_image_mixed, training=True)

            # computing the gradient penalty from the wasserstein-gp GAN
            gp_grads = gp_tape.gradient(fake_mixed_predicted, fake_image_mixed)
            gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

            # 3. now we actually do a feed forward with real and fake to the critic and check the loss
            # passing real and fake images through the critic
            real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)

            c_loss = self.discriminator_loss(real_predicted_patches, real_predicted_domain, real_domain,
                                             fake_predicted_patches, gradient_penalty)
            critic_total_loss, critic_real_loss, critic_fake_loss, critic_real_domain_loss, gp_regularization = c_loss

        # 4. apply the gradients to the critic weights
        discriminator_gradients = disc_tape.gradient(critic_total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        with tf.name_scope("discriminator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", critic_total_loss, step=step // update_steps)
                tf.summary.scalar("real_loss", critic_real_loss, step=step // update_steps)
                tf.summary.scalar("fake_loss", critic_fake_loss, step=step // update_steps)
                tf.summary.scalar("real_domain_loss", critic_real_domain_loss, step=step // update_steps)
                tf.summary.scalar("gradient_penalty", gp_regularization, step=step // update_steps)

        # TRAINING THE GENERATOR
        # ======================
        #
        # we only train the generator at every x discriminator update steps
        if step % self.discriminator_steps == 0:
            # 1. use previously selected random generator input (random source image/domain + random target domain)
            # target_domain = tf.random.uniform([batch_size, ], minval=0, maxval=self.config.number_of_domains,
            #                                   dtype="int32")
            # target_domain = tf.one_hot(target_domain, self.config.number_of_domains)
            # random_input = self.concat_image_and_domain(real_image, target_domain)
            # random_input, real_domain, real_image = self.select_random_input(domain_images)
            with tf.GradientTape() as gen_tape:
                # 2. feed forward the generator and critic
                fake_image = self.generator(random_input, training=True)
                fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)

                # 3. reconstruct the image to the original domain
                reconstruction_input = self.sampler.concat_image_and_domain(fake_image, real_domain)
                reconstructed_image = self.generator(reconstruction_input, training=True)

                # 4. calculate the loss
                g_loss = self.generator_loss(fake_predicted_patches, fake_predicted_domain, target_domain, real_image,
                                             reconstructed_image)
                generator_total_loss, generator_adversarial_loss, generator_domain_loss, \
                    generator_reconstruction_loss = g_loss

            # 5. update the generator weights using the gradients
            generator_gradients = gen_tape.gradient(generator_total_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            with tf.name_scope("generator"):
                with self.summary_writer.as_default():
                    tf.summary.scalar("total_loss", generator_total_loss, step=step // update_steps)
                    tf.summary.scalar("adversarial_loss", generator_adversarial_loss, step=step // update_steps)
                    tf.summary.scalar("domain_loss", generator_domain_loss, step=step // update_steps)
                    tf.summary.scalar("reconstruction_loss", generator_reconstruction_loss, step=step // update_steps)

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

            source_domain = tf.one_hot(tf.constant(source_domain)[tf.newaxis, ...], self.config.number_of_domains)
            target_domain = tf.one_hot(tf.constant(target_domain)[tf.newaxis, ...], self.config.number_of_domains)

            image_and_label = self.sampler.concat_image_and_domain(source_image[tf.newaxis, ...], target_domain)
            generated_image = self.generator(image_and_label, training=True)

            image_and_label = self.sampler.concat_image_and_domain(generated_image, source_domain)
            reconstructed_image = self.generator(image_and_label, training=True)

            images = [source_image, target_image, generated_image[0], reconstructed_image[0]]
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

        # cannot call show otherwise it flushes and empties the figure, sending to tensorboard
        # only a blank image... hence, let us just display the saved image
        # display.display(figure)
        # plt.show()

        return figure

    def evaluate_l1(self, real_images, fake_images):
        return tf.reduce_mean(tf.abs(fake_images - real_images))

    def initialize_random_examples_for_evaluation(self, train_ds, test_ds, num_images):
        number_of_domains = self.config.number_of_domains

        def initialize_random_examples_from_dataset(dataset):
            domain_images = next(iter(dataset.unbatch().take(num_images)))

            random_source_indices = tf.random.uniform(shape=[num_images], minval=0, maxval=number_of_domains,
                                                      dtype="int32")
            random_target_indices = tf.random.uniform(shape=[num_images], minval=0, maxval=number_of_domains,
                                                      dtype="int32")

            source_images = tf.gather(domain_images, random_source_indices)
            target_images = tf.gather(domain_images, random_target_indices)

            target_domains_one_hot = tf.one_hot(random_target_indices, number_of_domains)
            images_and_domains = self.sampler.concat_image_and_domain(source_images, target_domains_one_hot)
            return target_images, images_and_domains

        return dict({
            "train": initialize_random_examples_from_dataset(train_ds),
            "test": initialize_random_examples_from_dataset(test_ds.shuffle(self.config.test_size))
        })

    def generate_images_for_evaluation(self, example_indices_for_evaluation):
        generator = self.generator

        def generate_images_from_dataset(dataset_name):
            real_images, generator_input = example_indices_for_evaluation[dataset_name]
            fake_images = generator(generator_input, training=True)
            return real_images, fake_images

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def debug_discriminator_patches(self, batch, image_path):
        # generates the fake image and the discriminations of the real and fake
        domain_images = batch
        number_of_domains = self.config.number_of_domains
        batch_size, image_size = tf.shape(batch)[0], tf.shape(batch)[2]

        source_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=number_of_domains, dtype="int32")
        target_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=number_of_domains, dtype="int32")
        source_images = tf.gather(domain_images, source_indices, batch_dims=1)
        target_images = tf.gather(domain_images, target_indices, batch_dims=1)
        target_domains = tf.one_hot(target_indices, number_of_domains)
        generator_input = self.sampler.concat_image_and_domain(source_images, target_domains)

        fake_images = self.generator(generator_input, training=True)

        real_predicted_patches, real_predicted_domain = self.discriminator(target_images)
        fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_images)

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

    def generate_images_from_dataset(self, dataset, num_images=None):
        dataset = dataset.unbatch()
        if num_images is None:
            num_images = dataset.cardinality()

        dataset = list(dataset.take(num_images).as_numpy_iterator())

        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        # for each image in the dataset...
        for i, domain_images in enumerate(dataset):
            image_path = os.sep.join([base_image_path, f"{i}.png"])
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
                        target_domain = tf.one_hot(tf.constant(target_index)[tf.newaxis, ...], number_of_domains)
                        image_and_label = self.sampler.concat_image_and_domain(source_image[tf.newaxis, ...],
                                                                               target_domain)
                        generated_image = self.generator(image_and_label, training=True)
                        image = generated_image
                    plt.imshow(tf.squeeze(image) * 0.5 + 0.5)
                    plt.axis("off")
            plt.savefig(image_path, transparent=True)
            plt.close(fig)

        print(f"Generated {(i + 1) * number_of_domains} images in the test-images folder.")


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch):
        pass

    # def domain_index_one_hot(self, domain_index):
    #     number_of_domains = self.config.number_of_domains
    #     domain_one_hot = tf.one_hot(domain_index, number_of_domains, dtype="int32")
    #     return domain_one_hot

    def concat_image_and_domain(self, image, domain_one_hot_batched):
        image_size = self.config.image_size
        domain_one_hot_batched = domain_one_hot_batched[:, tf.newaxis, tf.newaxis, :]
        domain_as_channels = tf.tile(tf.cast(domain_one_hot_batched, "float32"), [1, image_size, image_size, 1])
        return tf.concat([image, domain_as_channels], axis=-1)


class SingleTargetSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)

    def sample(self, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        random_source_index = tf.random.uniform(shape=[], dtype="int32", minval=0, maxval=number_of_domains)
        random_source_index = tf.tile(random_source_index[tf.newaxis, ...], [batch_size,])
        transposed_batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        random_source_image = tf.gather(transposed_batch, random_source_index, axis=1, batch_dims=1)

        random_source_index = tf.one_hot(random_source_index, number_of_domains, dtype="int32")
        random_target_index = tf.random.uniform(shape=[], dtype="int32", minval=0, maxval=number_of_domains)
        random_target_index = tf.tile(random_target_index[tf.newaxis, ...], [batch_size,])
        random_target_index = tf.one_hot(random_target_index, number_of_domains, dtype="int32")

        image_and_domain = self.concat_image_and_domain(random_source_image, random_target_index)

        return image_and_domain, random_source_index, random_source_image, random_target_index


class MultiTargetSampler(ExampleSampler):
    def __init__(self, config):
        super().__init__(config)

    def sample(self, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = batch_shape[0], batch_shape[1]

        random_source_index = tf.random.uniform(shape=[batch_size], dtype="int32", minval=0, maxval=number_of_domains)
        transposed_batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        random_source_image = tf.gather(transposed_batch, random_source_index, axis=1, batch_dims=1)

        random_source_index = tf.one_hot(random_source_index, number_of_domains, dtype="int32")
        random_target_index = tf.random.uniform(shape=[batch_size], dtype="int32", minval=0, maxval=number_of_domains)
        random_target_index = tf.one_hot(random_target_index, number_of_domains, dtype="int32")

        image_and_domain = self.concat_image_and_domain(random_source_image, random_target_index)

        return image_and_domain, random_source_index, random_source_image, random_target_index
