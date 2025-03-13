import os
from functools import reduce

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model
from tqdm import tqdm

from utility import dataset_utils, io_utils
from remic_model import ConservativeDropoutSampler, UniformDropoutSampler
from side2side_model import S2SModel


class YamataModel(S2SModel):
    def __init__(self, config):
        super().__init__(config)
        self.sampler = ConservativeDropoutSampler(config)

        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()

        self.generator = self.inference_networks["generator"]
        self.discriminator = self.training_only_networks["discriminator"]

    def create_training_only_networks(self):
        return {
            "discriminator": build_discriminator()
        }

    def create_inference_networks(self):
        return {
            "generator": build_stochastic_unet()
        }

    def generator_loss(self, fake_output, source_images, fake_images):
        # Real/fake loss (fool the discriminator)
        adversarial_loss = self.mse(tf.ones_like(fake_output), fake_output)

        # MAE loss for supervision
        l1_loss = self.mae(source_images, fake_images)
        # l1_loss = tf.reduce_mean(tf.abs(source_images - fake_images))

        total_loss = (#adversarial_loss +
                      100. * l1_loss)
        return {
            "total": total_loss,
            "adversarial": adversarial_loss,
            "l1_forward": l1_loss,
            # "palette_loss": palette_loss
        }

    def discriminator_loss(self, real_predicted_patches, source_predicted_patches):
        adversarial_real = self.mse(tf.ones_like(real_predicted_patches), real_predicted_patches)
        adversarial_fake = self.mse(tf.zeros_like(source_predicted_patches), source_predicted_patches)

        total_loss = adversarial_real + adversarial_fake
        return {
            "total": total_loss,
            "real": adversarial_real,
            "fake": adversarial_fake
        }

    @tf.function
    def train_step(self, batch, step, evaluate_steps, t):
        # [d, b, s, s, c] = domain, batch, size, size, channels
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size, image_size, channels = self.config.number_of_domains, batch_shape[1], \
            batch_shape[2], batch_shape[4]

        # a. determines random domains to be dropped out
        real_images, input_keep_mask = self.sampler.sample(batch, t)
        source_images = real_images * input_keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        #-----------
        # print("input_keep_mask", input_keep_mask)
        # plot_5d_tensor_images(source_images)
        #-----------
        # source_images (shape=[b, d, s, s, c])
        # input_dropout_mask (shape=[b, d])
        # input_keep_mask = input_keep_mask[..., tf.newaxis, tf.newaxis, tf.newaxis]
        # input_keep_mask (shape=[b, d] -> [b, d, 1, 1, 1])

        # visible_source_images = source_images * input_keep_mask
        # visible_source_images (shape=[b, d, s, s, c])

        # Generate latent codes
        z = tf.random.normal((batch_size, 128))

        # Use a single GradientTape block for both generator and discriminator
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            # Generate fake images (4 poses per character)
            fake_images, masked_images = self.generator([source_images, z], training=True)
            # fake_images, masked_images = self.generator([source_images, input_keep_mask, z], training=True)

            # Discriminator outputs for real images
            # real_output = self.discriminator(source_images, training=True)

            # Discriminator outputs for fake images
            # fake_output = self.discriminator(fake_images, training=True)
            #
            # Discriminator losses
            # d_loss = self.discriminator_loss(real_output, fake_output)

            # Generator losses
            # g_loss = self.generator_loss(fake_output, source_images, fake_images)
            g_loss = self.generator_loss(tf.constant([0., 0., 0., 0.]), real_images, fake_images)

        # Compute discriminator gradients
        # disc_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        # self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Compute generator gradients
        gen_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        # Check for None gradients
        none_gradients = [var.name for var, grad in zip(self.generator.trainable_variables, gen_gradients) if grad is None]

        if none_gradients:
            print("Warning: None gradients found for the following variables:")
            for name in none_gradients:
                print(name)
        # else:
        #     print("No None gradients found.")

        #---------
        # plot_4d_tensor_images(masked_images)
        # plot_5d_tensor_images(fake_images)
        #---------

        summary_step = step // evaluate_steps
        with tf.name_scope("generator"):
            with self.summary_writer.as_default():
                tf.summary.scalar("total_loss", g_loss["total"], step=summary_step)
                tf.summary.scalar("adversarial_loss", g_loss["adversarial"], step=summary_step)
                tf.summary.scalar("l1_loss", g_loss["l1_forward"], step=summary_step)

        # with tf.name_scope("discriminator"):
        #     with self.summary_writer.as_default():
        #         tf.summary.scalar("total_loss", d_loss["total"], step=summary_step)
        #         tf.summary.scalar("real_loss", d_loss["real"], step=summary_step)
        #         tf.summary.scalar("fake_loss", d_loss["fake"], step=summary_step)

        # exit(0)

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

            source_images = tf.constant(source_images)
            keep_mask = tf.constant(keep_mask)

            generated_images, _ = self.generator.predict([source_images[tf.newaxis, ...],
                                                        tf.random.normal([1, 128])], batch_size=1, verbose=0)
            # generated_images, _ = self.generator.predict([source_images[tf.newaxis, ...], keep_mask[tf.newaxis, ...],
            #                                             tf.random.normal([1, 128])], batch_size=1, verbose=0)

            contents = [*tf.squeeze(source_images), *tf.squeeze(generated_images)]

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
            # return decoded_images
            fake_images = tf.gather(tf.transpose(decoded_images, [1, 0, 2, 3, 4]), possible_target_domain, axis=1,
                                    batch_dims=1)
            real_images = tf.gather(domain_images, possible_target_domain, batch_dims=1)
            return real_images, fake_images

        return {
            "train": generate_images_from_example_indices(example_indices_for_evaluation["train"]),
            "test": generate_images_from_example_indices(example_indices_for_evaluation["test"])
        }

    def generate_images_from_dataset(self, enumerated_dataset, step, num_images=None):
        raise NotImplementedError("This method is not implemented for this model")

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
        raise NotImplementedError("This method is not implemented for this model")


class MatMulLayer(layers.Layer):
    """Custom Keras layer to perform matrix multiplication."""
    def __init__(self, transpose_b=False, **kwargs):
        super(MatMulLayer, self).__init__(**kwargs)
        self.transpose_b = transpose_b

    def call(self, inputs):
        a, b = inputs
        return tf.matmul(a, b, transpose_b=self.transpose_b)

    # def compute_output_spec(self, inputs):
    #     """Compute the output shape of the layer."""
    #     a, b = inputs
    #     if self.transpose_b:
    #         output_shape = (a.shape[0], a.shape[1], b.shape[1])
    #     else:
    #         output_shape = (a.shape[0], a.shape[1], b.shape[2])
    #     return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(MatMulLayer, self).get_config()
        config.update({"transpose_b": self.transpose_b})
        return config


def build_self_attention_block(x, channels):
    """Self-attention block similar to SAGAN, using Keras layers."""
    batch, height, width, filters = tf.keras.backend.int_shape(x)

    # Query, Key, Value projections
    q = layers.Conv2D(channels // 8, 1, padding="same")(x)  # Query
    k = layers.Conv2D(channels // 8, 1, padding="same")(x)  # Key
    v = layers.Conv2D(channels, 1, padding="same")(x)  # Value

    # Reshape for attention computation
    q = layers.Reshape((height * width, channels // 8))(q)  # (batch_size, height * width, channels // 8)
    k = layers.Reshape((height * width, channels // 8))(k)  # (batch_size, height * width, channels // 8)
    v = layers.Reshape((height * width, channels))(v)  # (batch_size, height * width, channels)

    # Attention scores (dot product of Query and Key)
    matmul_layer = MatMulLayer(transpose_b=True)
    attn = matmul_layer([q, k])  # (batch_size, height * width, height * width)
    attn = layers.Softmax(axis=-1)(attn)  # Softmax over the last axis

    # Weighted sum of values
    out = matmul_layer([attn, v])  # (batch_size, height * width, channels)
    out = layers.Reshape((height, width, channels))(out)  # (batch_size, height, width, channels)

    # Add residual connection
    return layers.Add()([x, out])

# # Helper functions for the generator
# def build_self_attention_block(x, channels):
#     """Self-attention block similar to SAGAN, using Keras layers"""
#     batch, height, width, filters = tf.keras.backend.int_shape(x)
#
#     # Query, Key, Value projections
#     q = layers.Conv2D(channels // 8, 1, padding="same")(x)  # Query
#     k = layers.Conv2D(channels // 8, 1, padding="same")(x)  # Key
#     v = layers.Conv2D(channels, 1, padding="same")(x)  # Value
#
#     # Reshape for attention computation
#     q = layers.Reshape((height * width, channels // 8))(q)  # (batch_size, height * width, channels // 8)
#     k = layers.Reshape((height * width, channels // 8))(k)  # (batch_size, height * width, channels // 8)
#     v = layers.Reshape((height * width, channels))(v)  # (batch_size, height * width, channels)
#
#     # Transpose Key for dot product
#     k = tf.keras.layers.Permute((2, 1))(k)  # (batch_size, channels // 8, height * width)
#
#     # Attention scores (dot product of Query and Key)
#     attn = tf.keras.layers.Dot(axes=(2, 1))([q, k])  # (batch_size, height * width, height * width)
#     attn = layers.Softmax(axis=-1)(attn)  # Softmax over the last axis
#
#     # Weighted sum of values
#     out = tf.keras.layers.Dot(axes=(2, 1))([attn, v])  # (batch_size, height * width, channels)
#     out = layers.Reshape((height, width, channels))(out)  # (batch_size, height, width, channels)
#
#     # Add residual connection
#     return layers.Add()([x, out])


def residual_block(x, filters, initializer):
    """Residual block with group normalization"""
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1, epsilon=1e-6)(x)
    # x = layers.Activation("swish")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1, epsilon=1e-6)(x)
    #
    # # Skip connection
    # if shortcut.shape[-1] != filters:  # Adjust dimensions if necessary
    #     shortcut = layers.Conv2D(filters, 1, padding="same", kernel_initializer=initializer)(shortcut)

    x = layers.Add()([shortcut, x])
    # return layers.Activation("swish")(x)
    return layers.LeakyReLU(0.2)(x)


# Generator network
def build_stochastic_unet(latent_dim=128):
    """Generator network (U-Net) for image generation with additional residual blocks"""
    # Inputs: 4 images (64x64x4) and a mask (4,)
    images_input = layers.Input(shape=(4, 64, 64, 4), name="images_input")
    # mask_input = layers.Input(shape=(4,), name="mask_input")

    # Mask out missing images
    # masked_images = layers.Multiply()([images_input,
    #                                    layers.Reshape((4, 1, 1, 1))(mask_input)])
    masked_images = images_input

    # Combine images into a single tensor
    x = layers.Reshape((64, 64, 16))(masked_images)

    init = tf.random_normal_initializer(0., 0.02)
    # Encoder
    d1 = layers.Conv2D(64, 7, padding="same", kernel_initializer=init, use_bias=False)(x)
    d1 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d1)
    d1 = layers.ReLU()(d1)

    d1 = residual_block(d1, 64, init)
    # d1 = residual_block(d1, 64, init)  # Additional residual block
    d1 = build_self_attention_block(d1, 64)
    p1 = layers.MaxPooling2D(2)(d1)  # 32x32
    # p1 = layers.Conv2D(64, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d1)

    d2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, use_bias=False)(p1)
    d2 = residual_block(d2, 128, init)
    # d2 = residual_block(d2, 128, init)  # Additional residual block
    d2 = build_self_attention_block(d2, 128)
    p2 = layers.MaxPooling2D(2)(d2)  # 16x16
    # p2 = layers.Conv2D(128, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d2)
    d2_reduced = layers.Lambda(lambda x: x[..., :4])(d2)

    d3 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init, use_bias=False)(p2)
    d3 = residual_block(d3, 256, init)
    # d3 = residual_block(d3, 256, init)  # Additional residual block
    d3 = build_self_attention_block(d3, 256)
    p3 = layers.MaxPooling2D(2)(d3)  # 8x8
    # p3 = layers.Conv2D(256, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d3)
    d3_reduced = layers.Lambda(lambda x: x[..., :4])(d3)

    # # Bottleneck
    b = layers.Conv2D(512, 3, padding="same", kernel_initializer=init)(p3)
    b = residual_block(b, 512, init)
    # b = residual_block(b, 512, init)  # Additional residual block
    b = build_self_attention_block(b, 512)

    # Project latent variable to spatial dimensions
    z = layers.Input(shape=(latent_dim,), name="z_input")
    latent = layers.Dense(8 * 8 * 512)(z)
    latent = layers.Reshape((8, 8, 512))(latent)
    b = layers.Concatenate()([b, latent])  # Concatenate latent variable
    b = layers.Conv2D(512, 1, padding="same", kernel_initializer=init)(b)  # Mix features

    # # Decoder
    u1 = layers.UpSampling2D(2)(b)
    # u1 = layers.Conv2DTranspose(256, 2, strides=2, padding="same", kernel_initializer=init)(b)
    u1 = layers.Concatenate()([u1, d3])
    u1 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init)(u1)
    u1 = residual_block(u1, 256, init)
    # u1 = residual_block(u1, 256, init)  # Additional residual block
    u1 = build_self_attention_block(u1, 256)

    u2 = layers.UpSampling2D(2)(u1)
    # u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same", kernel_initializer=init)(u1)
    u2 = layers.Concatenate()([u2, d2])
    u2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init)(u2)
    u2 = residual_block(u2, 128, init)
    # u2 = residual_block(u2, 128, init)  # Additional residual block
    u2 = build_self_attention_block(u2, 128)

    u3 = layers.UpSampling2D(2)(u2)
    # u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same", kernel_initializer=init)(u2)
    u3 = layers.Concatenate()([u3, d1])
    u3 = layers.Conv2D(64, 3, padding="same", kernel_initializer=init)(u3)
    u3 = residual_block(u3, 64, init)
    # u3 = residual_block(u3, 64, init)  # Additional residual block
    u3 = build_self_attention_block(u3, 64)

    # Final output
    # output = layers.Conv2D(16, 3, padding="same", activation="tanh", kernel_initializer=init)(u3)  # Output in [-1, 1]
    # output = layers.Reshape((4, 64, 64, 4))(output)

    ending = layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(u3)
    outputs = [layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(ending)
               for _ in range(4)]
    outputs = [layers.GroupNormalization(groups=-1, epsilon=1e-6)(o) for o in outputs]
    outputs = [layers.LeakyReLU(0.2)(o) for o in outputs]
    outputs = [layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(o) for o in outputs]
    outputs = [layers.Conv2D(4, 1, padding="same", activation="tanh", kernel_initializer=init)(o) for o in outputs]
    output = layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)

    # return Model(inputs=[images_input, mask_input, z], outputs=[output, d2_reduced])
    return Model(inputs=[images_input, z], outputs=[output, d2_reduced])


def build_collalike_unet(latent_dim=128):
    """Generator network (U-Net) for image generation with additional residual blocks"""
    # Inputs: 4 images (64x64x4) and a mask (4,)
    images_input = layers.Input(shape=(4, 64, 64, 4), name="images_input")
    # mask_input = layers.Input(shape=(4,), name="mask_input")

    # Mask out missing images
    # masked_images = layers.Multiply()([images_input,
    #                                    layers.Reshape((4, 1, 1, 1))(mask_input)])
    masked_images = images_input

    # Combine images into a single tensor
    x = layers.Reshape((64, 64, 16))(masked_images)

    init = tf.random_normal_initializer(0., 0.02)
    # Encoder
    d1 = layers.Conv2D(64, 3, padding="same", kernel_initializer=init, use_bias=False)(x)
    d1 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d1)
    d1 = layers.LeakyReLU(0.2)(d1)
    d1 = layers.Conv2D(64, 3, padding="same", kernel_initializer=init, use_bias=False)(d1)
    d1 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d1)
    d1 = layers.LeakyReLU(0.2)(d1)

    p1 = layers.Conv2D(64, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d1)

    d2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, use_bias=False)(p1)
    d2 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d2)
    d2 = layers.LeakyReLU(0.2)(d2)
    d2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, use_bias=False)(d2)
    d2 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d2)
    d2 = layers.LeakyReLU(0.2)(d2)

    p2 = layers.Conv2D(128, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d2)
    d2_reduced = layers.Lambda(lambda x: x[..., :4])(d2)

    d3 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init, use_bias=False)(p2)
    d3 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d3)
    d3 = layers.LeakyReLU(0.2)(d3)
    d3 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init, use_bias=False)(d3)
    d3 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d3)
    d3 = layers.LeakyReLU(0.2)(d3)

    p3 = layers.Conv2D(256, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d3)

    d4 = layers.Conv2D(512, 3, padding="same", kernel_initializer=init, use_bias=False)(p3)
    d4 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d4)
    d4 = layers.LeakyReLU(0.2)(d4)
    d4 = layers.Conv2D(512, 3, padding="same", kernel_initializer=init, use_bias=False)(d4)
    d4 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d4)
    d4 = layers.LeakyReLU(0.2)(d4)

    p4 = layers.Conv2D(1024, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d4)

    d5 = layers.Conv2D(1024, 3, padding="same", kernel_initializer=init, use_bias=False)(p4)
    d5 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d5)
    d5 = layers.LeakyReLU(0.2)(d5)
    d5 = layers.Conv2D(1024, 3, padding="same", kernel_initializer=init, use_bias=False)(d5)
    d5 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(d5)
    d5 = layers.LeakyReLU(0.2)(d5)

    up5 = layers.Conv2DTranspose(512, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(d5)
    up5 = layers.Concatenate()([up5, d4])

    up4 = layers.Conv2D(512, 3, padding="same", kernel_initializer=init, use_bias=False)(up5)
    up4 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up4)
    up4 = layers.LeakyReLU(0.2)(up4)
    up4 = layers.Conv2D(512, 3, padding="same", kernel_initializer=init, use_bias=False)(up4)
    up4 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up4)
    up4 = layers.LeakyReLU(0.2)(up4)

    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(up4)
    up3 = layers.Concatenate()([up3, d3])
    up3 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init, use_bias=False)(up3)
    up3 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up3)
    up3 = layers.LeakyReLU(0.2)(up3)
    up3 = layers.Conv2D(256, 3, padding="same", kernel_initializer=init, use_bias=False)(up3)
    up3 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up3)
    up3 = layers.LeakyReLU(0.2)(up3)

    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(up3)
    up2 = layers.Concatenate()([up2, d2])
    up2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, use_bias=False)(up2)
    up2 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up2)
    up2 = layers.LeakyReLU(0.2)(up2)
    up2 = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, use_bias=False)(up2)
    up2 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up2)
    up2 = layers.LeakyReLU(0.2)(up2)

    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same", kernel_initializer=init, use_bias=False)(up2)
    up1 = layers.Concatenate()([up1, d1])
    up1 = layers.Conv2D(64, 3, padding="same", kernel_initializer=init, use_bias=False)(up1)
    up1 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up1)
    up1 = layers.LeakyReLU(0.2)(up1)
    up1 = layers.Conv2D(64, 3, padding="same", kernel_initializer=init, use_bias=False)(up1)
    up1 = layers.GroupNormalization(groups=-1, epsilon=1e-6)(up1)
    up1 = layers.LeakyReLU(0.2)(up1)

    ending = layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(up1)
    outputs = [layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(ending)
               for _ in range(4)]
    outputs = [layers.GroupNormalization(groups=-1, epsilon=1e-6)(o) for o in outputs]
    outputs = [layers.LeakyReLU(0.2)(o) for o in outputs]
    outputs = [layers.Conv2D(16, 3, padding="same", kernel_initializer=init)(o) for o in outputs]
    outputs = [layers.Conv2D(4, 1, padding="same", activation="tanh", kernel_initializer=init)(o) for o in outputs]
    output = layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)
    z = layers.Input(shape=(latent_dim,), name="z_input")

    return Model(inputs=[images_input, z], outputs=[output, d2_reduced])


def build_discriminator(image_shape=(64, 64, 4)):
    """Conditional discriminator network for GAN training"""
    inputs = layers.Input(shape=(4, 64, 64, 4))  # Input shape: (batch_size, 4, 64, 64, 4)

    # Shared convolutional layers for feature extraction
    def shared_conv_layers(x):
        x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Flatten()(x)
        return x

    # Extract features for each pose
    pose_features = []
    for i in range(4):
        pose = inputs[:, i, :, :, :]  # Extract the i-th pose
        features = shared_conv_layers(pose)  # Extract features using shared layers
        pose_features.append(features)

    # Concatenate features from all poses
    combined_features = layers.Concatenate()(pose_features)

    # Dense layers for joint decision
    x = layers.Dense(512, activation='relu')(combined_features)
    x = layers.Dense(128, activation='relu')(x)

    # Real/fake score (linear activation for LSGAN)
    real_fake_score = layers.Dense(1, activation='linear')(x)

    return Model(inputs, real_fake_score)


def plot_5d_tensor_images(tensor):
    """
    Plots a tensor with shape [batch, domains, image_size, image_size, channels] and values in the [-1, 1] domain.

    Args:
        tensor: A tensor with shape [batch, domains, image_size, image_size, channels].
    """
    batch, domains, image_size, _, channels = tensor.shape
    fig, axes = plt.subplots(batch, domains, figsize=(domains * 2, batch * 2))

    for i in range(batch):
        for j in range(domains):
            image = tensor[i, j]
            image = (image + 1) / 2  # Convert from [-1, 1] to [0, 1] for plotting
            if channels == 1:
                image = tf.squeeze(image, axis=-1)  # Remove the channel dimension if it's 1
            axes[i, j].imshow(image)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_4d_tensor_images(tensor):
    """
    Plots a tensor with shape [batch, image_size, image_size, channels] and values in the [-1, 1] domain.

    Args:
        tensor: A tensor with shape [batch, image_size, image_size, channels].
    """
    batch, image_size, _, channels = tensor.shape
    fig, axes = plt.subplots(batch, 1, figsize=(1 * 2, batch * 2))

    for i in range(batch):
        image = tensor[i]
        image = (image + 1) / 2  # Convert from [-1, 1] to [0, 1] for plotting
        if channels == 1:
            image = tf.squeeze(image, axis=-1)  # Remove the channel dimension if it's 1
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# python train.py yamata --rm2k --model-name yamata --lr 0.00005 --steps 5000 --evaluate-steps 100 --lr 0.00001