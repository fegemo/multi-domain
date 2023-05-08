import tensorflow as tf
from tensorflow.keras import layers
from configuration import *
from tensorflow_addons import layers as tfalayers


def unet_downsample(filters, size, apply_batchnorm=True, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()

    result.add(layers.Conv2D(
        filters,
        size,
        strides=2,
        padding="same",
        kernel_initializer=init,
        use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())

    return result


def unet_upsample(filters, size, apply_dropout=False, init=tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def PatchDiscriminator(input_channels):
    initializer = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, input_channels], name="source_image")
    target_image = layers.Input(shape=[IMG_SIZE, IMG_SIZE, input_channels], name="target_image")

    x = layers.concatenate([target_image, source_image])  # (batch_size, 64, 64, channels*2)
    down = unet_downsample(64, 4, False)(x)  # (batch_size, 32, 32,         64)
    last = layers.Conv2D(1, 4, padding="same",  # (batch_size, 32, 32,          1)
                         kernel_initializer=initializer)(down)

    return tf.keras.Model(inputs=[target_image, source_image], outputs=last, name="patch-disc")


def UnetGenerator(input_channels, output_channels, last_activation):
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, input_channels])  # (batch_size, 64, 64, 4 or 1)

    down_stack = [
        unet_downsample(64, 4, apply_batchnorm=False, init=init),  # (batch_size, 32, 32,   64)
        unet_downsample(128, 4, init=init),  # (batch_size, 16, 16,  128)
        unet_downsample(256, 4, init=init),  # (batch_size,  8,  8,  256)
        unet_downsample(512, 4, init=init),  # (batch_size,  4,  4,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  2,  2,  512)
        unet_downsample(512, 4, init=init),  # (batch_size,  1,  1,  512)
    ]

    up_stack = [
        unet_upsample(512, 4, apply_dropout=True, init=init),  # (batch_size,  2,  2, 1024)
        unet_upsample(512, 4, apply_dropout=True, init=init),  # (batch_size,  4,  4, 1024)
        unet_upsample(256, 4, apply_dropout=True, init=init),  # (batch_size,  8,  8,  512)
        unet_upsample(128, 4, init=init),  # (batch_size, 16, 16,  256)
        unet_upsample(64, 4, init=init),  # (batch_size, 32, 32,  128)
        unet_upsample(32, 4, init=init),  # (batch_size, 64, 64,   36)
    ]

    last = layers.Conv2D(output_channels, 4,  # (batch_size, 64, 64, 4 or 256)
                         padding="same",
                         kernel_initializer=init,
                         activation=last_activation)

    x = inputs

    # executing down sampling and saving the skip-connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ignores the last skip and reverses them
    skips = list(reversed(skips[:-1]))

    # up samples and concatenates the partial results from skips
    for up, skip in zip(up_stack, [*skips, inputs]):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="unet-gen")


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


def stargan_resnet_discriminator(number_of_domains):
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
    classification = layers.Conv2D(number_of_domains, kernel_size=full_kernel_size, strides=1, kernel_initializer=init,
                                   use_bias=False)(x)
    classification = layers.Reshape((number_of_domains,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="StarGANDiscriminator")


def stargan_resnet_generator(image_size, output_channels, number_of_domains):
    init = tf.random_normal_initializer(0., 0.02)

    source_image = layers.Input(shape=[image_size, image_size, output_channels + number_of_domains])
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

    x = layers.Conv2D(output_channels, kernel_size=7, strides=1, padding="same", kernel_initializer=init,
                      use_bias=False)(x)
    activation = layers.Activation("tanh", name="generated_image")(x)

    return tf.keras.Model(inputs=source_image, outputs=activation, name="StarGANGenerator")
