import os
from abc import ABC, abstractmethod

import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

import dataset_utils
import io_utils
from networks import collagan_affluent_generator, collagan_original_discriminator
from side2side_model import S2SModel
from keras_utils import NParamsSupplier


class CollaGANModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)

        self.lambda_l1 = config.lambda_l1 or config.lambda_l1
        self.lambda_l1_backward = config.lambda_l1_backward or config.lambda_l1
        self.lambda_domain = config.lambda_domain
        self.lambda_ssim = config.lambda_ssim

        if config.input_dropout:
            self.sampler = InputDropoutSampler(config)
        else:
            self.sampler = SimpleSampler(config)

        if config.cycled_source_replacer in ["", "dropout"]:
            self.cycled_source_replacer = DroppedOutCycledSourceReplacer(config)
        else:
            self.cycled_source_replacer = ForwardOnlyCycledSourceReplacer(config)

        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.generator_supplier = NParamsSupplier(2)

    def create_generator(self):
        config = self.config
        if config.generator in ["colla", ""]:
            return collagan_affluent_generator(config.number_of_domains, config.image_size, config.output_channels)
        else:
            raise ValueError(f"The provided {config.generator} type for generator has not been implemented.")

    def create_discriminator(self):
        config = self.config
        if config.discriminator in ["colla", ""]:
            return collagan_original_discriminator(config.number_of_domains, config.image_size, config.output_channels)
        else:
            raise ValueError(f"The provided {config.discriminator} type for discriminator has not been implemented.")

    def generator_loss(self, fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                       cycled_images, source_images_5d,
                       fake_predicted_domain, cycle_predicted_domain, target_domain,
                       input_dropout_mask, batch_shape):
        number_of_domains = batch_shape[0]
        batch_size, image_size, channels = batch_shape[1], batch_shape[2], batch_shape[4]
        number_of_domains_float = tf.cast(number_of_domains, tf.float32)
        source_images = tf.reshape(source_images_5d, [batch_size * number_of_domains, image_size, image_size, channels])
        cycled_images_5d = tf.reshape(cycled_images, [batch_size, number_of_domains, image_size, image_size, channels])
        input_dropout_mask_1d = tf.reshape(input_dropout_mask, [batch_size * number_of_domains])

        # adversarial (lsgan) loss
        adversarial_forward__loss = tf.reduce_mean(tf.math.squared_difference(fake_predicted_patches, 1.))
        adversarial_backward_loss = tf.reduce_mean(tf.math.squared_difference(cycled_predicted_patches, 1.)) * \
                                    number_of_domains_float
        adversarial_loss = (adversarial_forward__loss + adversarial_backward_loss) / \
                           (number_of_domains_float + 1.)

        # l1 (forward, backward)
        l1_forward__loss = tf.reduce_mean(tf.abs(real_image - fake_image))
        l1_backward_loss = tf.reduce_mean(
            tf.reduce_sum(
                # mean of pixel l1s per image, but 0 for dropped out input images
                tf.reduce_mean(tf.abs(source_images_5d - cycled_images_5d), axis=[2, 3, 4]) * input_dropout_mask,
                axis=1),
            axis=0)

        # ssim loss (forward, backward)
        ssim_forward_ = tf.image.ssim(fake_image + 1., real_image + 1., 2)
        ssim_backward = tf.image.ssim(cycled_images + 1., source_images + 1., 2) * input_dropout_mask_1d
        # ssim_forward_ (shape=[b,])
        # ssim_backward (shape=[b*d,])
        ssim_forward__loss = tf.reduce_mean(-tf.math.log((1. + ssim_forward_) / 2.))
        ssim_backward_loss = tf.reduce_mean(tf.reduce_sum(-tf.math.log((1. + ssim_backward) / 2.)))
        ssim_loss = (ssim_forward__loss + ssim_backward_loss * number_of_domains_float) / (number_of_domains_float + 1.)

        # domain classification loss (forward, backward)
        forward__domain = tf.one_hot(target_domain, number_of_domains)
        backward_domain = tf.tile(tf.one_hot(tf.range(number_of_domains), number_of_domains), [batch_size, 1])
        backward_predicted_domain = cycle_predicted_domain
        # forward__domain (shape=[b, d])
        # backward_domain (shape=[b*d, d])

        classification_forward__loss = self.cce(forward__domain, fake_predicted_domain)
        classification_backward_loss = self.cce(backward_domain, backward_predicted_domain) * number_of_domains_float
        classification_loss = (classification_forward__loss + classification_backward_loss) / \
                              (number_of_domains_float + 1.)

        # observation: ssim loss uses only the backward (cycled) images... that's on the colla's code and paper
        total_loss = adversarial_loss + \
                     self.lambda_l1 * l1_forward__loss + self.lambda_l1_backward * l1_backward_loss + \
                     self.lambda_ssim * ssim_backward_loss + \
                     self.lambda_domain * classification_loss

        return {"total": total_loss, "adversarial": adversarial_loss, "l1_forward": l1_forward__loss,
                "l1_backward": l1_backward_loss, "ssim": ssim_loss, "domain": classification_loss}

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

        total_loss = adversarial_loss + \
                     self.lambda_domain * domain_loss
        return {"total": total_loss, "real": adversarial_real, "fake": adversarial_fake, "domain": domain_loss}

    def get_cycled_images_input(self, domain_images, forward_target_domain, input_dropout_mask, fake_image,
                                batch_shape):
        """
        Returns a list of tensors that represent the input for the generator to create the cycle images
        :param domain_images: batch images for all domains (shape=[b, d, s, s, c])
        :param forward_target_domain: batched target domain index (shape=[b])
        :param input_dropout_mask: mask for which domain images have been dropped out (due to input dropout and being
        the target domain) (shape=[b, d], with 0s for dropped out images)
        :param fake_image: batched generated image (shape=[b, s, s, c])
        :param batch_shape: tuple representing the shape of the batch
        :return: a list of tensors that can be used as input to the generator so cycle images get created
        """
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], batch_shape[2], \
            batch_shape[4]
        backward_target_domain = tf.range(number_of_domains)
        backward_target_domain = tf.tile(backward_target_domain[tf.newaxis, ...], [batch_size, 1])
        # backward_target_domain (shape=[b, d]) with an index per image and domain in the batch

        backward_target_domain_mask = tf.one_hot(backward_target_domain, number_of_domains, on_value=0., off_value=1.)
        backward_target_domain_mask = backward_target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # backward_target_domain_mask (shape=[b, d_target, d, 1, 1, 1]) with 0s for images that must be suppressed
        # (as they are the target)

        # a. repeat the domain images once for each domain, so we can later have an input set with a
        # zeroed backward target for each domain
        repeated_domain_images = tf.tile(tf.expand_dims(domain_images, 1), [1, number_of_domains, 1, 1, 1, 1])
        # repeated_domain_images (shape=[b, d_target, d, s, s, c]

        # b. replace the original forward target image with the generated fake image
        # forward_target_domain_mask = tf.one_hot(forward_target_domain, number_of_domains,
        #                                         dtype=tf.bool, on_value=True, off_value=False)
        # forward_target_domain_mask = forward_target_domain_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape

        # b. replace the original forward target image with the generated fake image, plus the ones that have
        # been dropped out. it can be only the forward target (--cycled-source-replacer == "forward") or all the
        # source images that have been dropped out due to input dropout (--cycled-source-replacer == "dropout")
        # the Colla's paper does not specify what it does, but the source code¹ uses the "dropout" option
        # ¹ https://github.com/jongcye/CollaGAN_CVPR/blob/master/model/CollaGAN_fExp8.py#L99
        cycled_source_replacement_mask = self.cycled_source_replacer.replace(forward_target_domain, input_dropout_mask)
        fake_image = fake_image[:, tf.newaxis, tf.newaxis, ...]
        # fake_image becomes shape=[b, 1, 1, s, s, c], so it can be broadcast together with repeated_domain_images

        fake_replaced_target_domain_images = tf.where(cycled_source_replacement_mask, fake_image,
                                                      repeated_domain_images)
        # fake_replaced_target_domain_images (shape=[b, d, d, s, s, c])

        # c. zero out the images that are the backwards cyclical target
        zeroed_retarget_domain_images = fake_replaced_target_domain_images * backward_target_domain_mask

        # list of:
        # - input images with shape [b, d, d, s, s, c]
        # - input target domain with shape [b, d]
        # - TODO more to come??

        zeroed_retarget_domain_images = tf.reshape(zeroed_retarget_domain_images, [
            batch_size * number_of_domains, number_of_domains, image_size, image_size, channels
        ])
        backward_target_domain = tf.reshape(backward_target_domain, [batch_size * number_of_domains])

        return self.generator_supplier(zeroed_retarget_domain_images, backward_target_domain)

    @tf.function
    def train_step(self, batch, step, update_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = batch_shape[0], batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # 1. select a random target domain with a subset of the images as input
        domain_images, target_domain, input_dropout_mask = self.sampler.sample(batch)
        # domain_images (shape=[b, d, s, s, c])
        # target_domain (shape=[b,])
        # input_dropout_mask (shape=[b, d]), with 0s for images that should be dropped out

        # dropped_input_images contains, for each domain, an image or zeros with same shape (dropped out)
        dropped_input_image = domain_images * input_dropout_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # dropped_input_image (shape=[b, d, s, s, c], but with all 0s for dropped images)

        # real_image is the target for each example in the batch
        real_image = tf.gather(domain_images, target_domain, batch_dims=1)

        with tf.GradientTape(persistent=True) as tape:
            # 1. generate a batch of fake images
            # shape=[b, d, s, s, c]
            generator_input = self.generator_supplier(dropped_input_image, target_domain)
            # shape=[b, s, s, c]
            fake_image = self.generator(generator_input, training=True)

            # 2. generate a batch of cycled images (back to their source domain)
            cycled_generator_input = self.get_cycled_images_input(domain_images, target_domain, input_dropout_mask,
                                                                  fake_image, batch_shape)
            # cycled_generator_input (list of [shape=[b*d, d, s, s, c], shape=[b*d]])
            cycled_images = self.generator(cycled_generator_input, training=True)
            # cycled_images (shape=[b*d, s, s, c])

            # 3. discriminate the real (target) and fake images, then the cycled ones and the source (to train the disc)
            real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
            fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)
            # xxxx_predicted_patches (shape=[b, 1, 1, 1])
            # xxxx_predicted_domain  (shape=[b, d] -> logits)

            cycled_predicted_patches, cycled_predicted_domain = self.discriminator(cycled_images, training=True)
            # cycled_predicted_patches (shape=[b*d, 1, 1, 1])
            # cycled_predicted_domain  (shape=[b*d, d] -> logits)

            source_predicted_patches, source_predicted_domain = self.discriminator(
                tf.reshape(domain_images, [-1, image_size, image_size, channels]), training=True)
            # source_predicted_patches (shape=[b*d, 1, 1, 1])
            # source_predicted_domain  (shape=[b*d, d] -> logits)

            # 4. calculate loss terms for the generator
            g_loss = self.generator_loss(fake_predicted_patches, cycled_predicted_patches, fake_image, real_image,
                                         cycled_images, domain_images, fake_predicted_domain,
                                         cycled_predicted_domain,
                                         target_domain, input_dropout_mask, batch_shape)

            # 5. calculate loss terms for the discriminator
            d_loss = self.discriminator_loss(source_predicted_patches, cycled_predicted_patches,
                                             source_predicted_domain, real_predicted_patches, fake_predicted_patches,
                                             batch_shape)

        generator_gradients = tape.gradient(g_loss["total"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        discriminator_gradients = tape.gradient(d_loss["total"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        summary_step = step // update_steps
        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", g_loss["total"], step=summary_step)
                tf.summary.scalar("adversarial_loss", g_loss["adversarial"], step=summary_step)
                tf.summary.scalar("domain_loss", g_loss["domain"], step=summary_step)
                tf.summary.scalar("ssim_loss", g_loss["ssim"], step=summary_step)
                tf.summary.scalar("l1_forward_loss", g_loss["l1_forward"], step=summary_step)
                tf.summary.scalar("l1_backward_loss", g_loss["l1_backward"], step=summary_step)

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

            real_image = domain_images[target_domain]
            domain_images = tf.constant(domain_images)

            # this zeroes out the target image:
            target_domain_mask = tf.one_hot(target_domain, number_of_domains, on_value=0., off_value=1.)
            domain_images *= target_domain_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]

            fake_image = self.generator(
                self.generator_supplier(
                    tf.expand_dims(domain_images, 0),
                    tf.expand_dims(target_domain, 0)
                ),
                training=True)

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

                plt.imshow(images[j] * 0.5 + 0.5)
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

            fake_image = generator(self.generator_supplier(domain_images, target_domain), training=True)
            return target_image, fake_image

        return dict({
            "train": generate_images_from_dataset("train"),
            "test": generate_images_from_dataset("test")
        })

    def generate_images_from_dataset(self, dataset, step, num_images=None):
        dataset = dataset.unbatch()
        if num_images is None:
            num_images = dataset.cardinality()

        dataset = list(dataset.take(num_images).as_numpy_iterator())

        base_image_path = self.get_output_folder("test-images")

        io_utils.delete_folder(base_image_path)
        io_utils.ensure_folder_structure(base_image_path)

        number_of_domains = self.config.number_of_domains
        # for each image i in the dataset...
        for i, domain_images in enumerate(tqdm(dataset, total=len(dataset))):
            # for each number m of missing domains [1 to d[
            for m in range(1, number_of_domains):
                image_path = os.sep.join([base_image_path, f"{i}_at_step_{step}_missing_{m}.png"])
                fig = plt.figure(figsize=(4 * number_of_domains, 4 * number_of_domains))
                plt.suptitle(f"Missing {m} image(s)", fontdict={"size": 20})
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
                        self.generator_supplier(dropped_input_image, target_domain), training=True)

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

                        plt.imshow(image * 0.5 + 0.5)
                        plt.axis("off")

                plt.savefig(image_path, transparent=True)
                plt.close(fig)

        print(f"Generated {(i + 1) * number_of_domains * (number_of_domains - 1)} images in the test-images folder.")

    def debug_discriminator_output(self, batch, image_path):
        # batch (shape=(d, b, s, s, c))
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size = batch_shape[0], batch_shape[1], batch_shape[2]

        domain_images, target_domain, _ = self.sampler.sample(batch)
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
        fake_image = self.generator(self.generator_supplier(forward__input_image, target_domain), training=True)

        forward__target_mask = tf.cast(1. - forward__target_mask, tf.bool)
        backward_input_image = domain_images * backward_target_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        backward_input_image = tf.where(
            forward__target_mask[..., tf.newaxis, tf.newaxis, tf.newaxis],
            fake_image[:, tf.newaxis, ...],
            backward_input_image)

        back_image = self.generator(self.generator_supplier(backward_input_image, source_domain), training=True)

        real_predicted_patches, real_predicted_domain = self.discriminator(real_image, training=True)
        fake_predicted_patches, fake_predicted_domain = self.discriminator(fake_image, training=True)
        back_predicted_patches, back_predicted_domain = self.discriminator(back_image, training=True)

        real_predicted_mean = tf.reduce_mean(real_predicted_patches, axis=[1, 2, 3])
        fake_predicted_mean = tf.reduce_mean(fake_predicted_patches, axis=[1, 2, 3])
        back_predicted_mean = tf.reduce_mean(back_predicted_patches, axis=[1, 2, 3])

        real_predicted_domain = tf.math.argmax(real_predicted_domain, axis=1, output_type=tf.int32)
        fake_predicted_domain = tf.math.argmax(fake_predicted_domain, axis=1, output_type=tf.int32)
        back_predicted_domain = tf.math.argmax(back_predicted_domain, axis=1, output_type=tf.int32)

        # lsgan yields an unbounded real number, which should be 1 for real images and 0 for fake
        # but we need to provide them in the [0, 1] range
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
                        " (", self.config.domains[real_predicted_domain[i]], ")"]).numpy().decode("utf-8")

                elif titles[j] == "Disc. Gen.":
                    subplot_title = tf.strings.join([
                        titles[j],
                        " ",
                        tf.strings.as_string(fake_predicted_mean[i], precision=3),
                        " (", self.config.domains[fake_predicted_domain[i]], ")"]).numpy().decode("utf-8")

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
                        " (", self.config.domains[back_predicted_domain[i]], ")"]).numpy().decode("utf-8")

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


class ExampleSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def sample(self, batch):
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

    def sample(self, batch):
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
        random_number_of_inputs_to_drop = tf.random.uniform(shape=[batch_size],
                                                            maxval=tf.shape(dropout_null_list_for_target[0])[0],
                                                            dtype="int32")
        dropout_null_list_for_target_and_number_of_inputs = tf.gather(dropout_null_list_for_target,
                                                                      random_number_of_inputs_to_drop,
                                                                      batch_dims=1)
        # dropout_null_list_for_target_and_number_of_inputs (shape=[b, ?, d])
        random_permutation_index = tf.random.uniform(shape=[batch_size],
                                                     maxval=tf.shape(dropout_null_list_for_target_and_number_of_inputs)[
                                                         0], dtype="int32")
        input_dropout_mask = tf.gather(dropout_null_list_for_target_and_number_of_inputs,
                                       random_permutation_index,
                                       batch_dims=1)
        # input_dropout_mask (shape=[b, d])
        input_dropout_mask = tf.where(input_dropout_mask, 0., 1.)

        return batch, target_domain_index, input_dropout_mask


class SimpleSampler(ExampleSampler):
    def sample(self, batch):
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

    def replace(self, forward_target_domain, input_dropout_mask):
        number_of_domains = self.config.number_of_domains
        forward_target_domain_mask = tf.one_hot(forward_target_domain, number_of_domains,
                                                dtype=tf.bool, on_value=True, off_value=False)
        forward_target_domain_mask = forward_target_domain_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape
        return forward_target_domain_mask


class DroppedOutCycledSourceReplacer(CycledSourceReplacer):
    def __init__(self, config):
        super().__init__(config)

    def replace(self, forward_target_domain, input_dropout_mask):
        inverted_input_dropout_mask = tf.logical_not(tf.cast(input_dropout_mask, tf.bool))
        inverted_input_dropout_mask = inverted_input_dropout_mask[:, tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
        # the mask becomes of shape [b, 1, d, 1, 1, 1], which can be broadcast to the repeated_domain_images' shape

        return inverted_input_dropout_mask
