import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons import layers as tfalayers

import keras_utils


def resblock(x, filters, kernel_size, init):
    original_x = x

    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.Add()([original_x, x])

    # the StarGAN official implementation skips this last activation of the resblock
    # https://github.com/yunjey/stargan/blob/master/model.py
    # x = layers.ReLU()(x)
    return x


def stargan_resnet_discriminator(number_of_domains, image_size, output_channels):
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[image_size, image_size, output_channels], name="source_image")
    x = source_image

    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    filters = 64
    downsampling_blocks = 5  # 128, 256, 512, 1024, 2048
    for i in range(downsampling_blocks):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = layers.LeakyReLU(0.01)(x)

    # 2x2 patches output (2x2x1)
    patches = layers.Conv2D(1, kernel_size=3, strides=1, padding="same", kernel_initializer=init, use_bias=False,
                            name="discriminator_patches")(x)

    # domain classifier output (1 x 1 x domain)
    full_kernel_size = image_size // (2 ** downsampling_blocks)
    classification = layers.Conv2D(number_of_domains, kernel_size=full_kernel_size, strides=1, kernel_initializer=init,
                                   use_bias=False)(x)
    classification = layers.Reshape((number_of_domains,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="StarGANDiscriminator")


def stargan_resnet_generator(image_size, output_channels, number_of_domains):
    init = tf.random_normal_initializer(0., 0.02)

    source_image_input = layers.Input(shape=[image_size, image_size, output_channels], name="source_image")
    target_domain_input = layers.Input(shape=[1], name="target_domain")
    target_domain = layers.CategoryEncoding(num_tokens=number_of_domains, output_mode="one_hot")(target_domain_input)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)

    x = layers.Concatenate(axis=-1)([source_image_input, target_domain])

    filters = 64
    x = layers.Conv2D(filters, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
    x = layers.ReLU()(x)

    # downsampling blocks: 128, then 256
    for i in range(2):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    # bottleneck blocks
    for i in range(6):
        x = resblock(x, filters, 3, init)

    # upsampling blocks: 128, then 64
    for i in range(2):
        filters /= 2
        x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init,
                                   use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    x = layers.Conv2D(output_channels, kernel_size=7, strides=1, padding="same", kernel_initializer=init,
                      use_bias=False)(x)
    activation = layers.Activation("tanh", name="generated_image")(x)

    return tf.keras.Model(inputs=[source_image_input, target_domain_input], outputs=activation, name="StarGANGenerator")
