import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons import layers as tfalayers

OUTPUT_CHANNELS = 4
IMG_SIZE = 64
NUMBER_OF_DOMAINS = 4

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


def StarGANDiscriminator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name="source_image")
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

    # domain classifier output (1x1xdomain)
    full_kernel_size = IMG_SIZE // (2 ** downsampling_blocks)
    classification = layers.Conv2D(NUMBER_OF_DOMAINS, kernel_size=full_kernel_size, strides=1, kernel_initializer=init,
                                   use_bias=False)(x)
    classification = layers.Reshape((NUMBER_OF_DOMAINS,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="StarGANDiscriminator")


def StarGANGenerator():
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS + NUMBER_OF_DOMAINS])
    x = source_image

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

    x = layers.Conv2D(OUTPUT_CHANNELS, kernel_size=7, strides=1, padding="same", kernel_initializer=init,
                      use_bias=False)(x)
    activation = layers.Activation("tanh", name="generated_image")(x)

    return tf.keras.Model(inputs=source_image, outputs=activation, name="StarGANGenerator")
