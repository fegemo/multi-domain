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


def stargan_resnet_discriminator(number_of_domains, image_size, output_channels, receive_source_image):
    init = tf.random_normal_initializer(0., 0.02)

    real_or_fake_image = layers.Input(shape=[image_size, image_size, output_channels], name="real_or_fake_image")
    inputs = [real_or_fake_image]
    if receive_source_image:
        source_image = layers.Input(shape=[image_size, image_size, output_channels], name="source_image")
        inputs += [source_image]

    # downsampling blocks (1 less than StarGAN b/c our input is star/2)
    x = layers.Concatenate(axis=-1)(inputs)
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

    return tf.keras.Model(inputs=inputs, outputs=[patches, classification], name="StarGANDiscriminator")


def stargan_resnet_generator(image_size, output_channels, number_of_domains, receive_source_domain):
    init = tf.random_normal_initializer(0., 0.02)

    source_image_input = layers.Input(shape=[image_size, image_size, output_channels], name="source_image")
    target_domain_input = layers.Input(shape=[1], name="target_domain")
    target_domain = layers.CategoryEncoding(num_tokens=number_of_domains, output_mode="one_hot")(target_domain_input)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)

    inputs = [source_image_input, target_domain_input]
    if not receive_source_domain:
        x = layers.Concatenate(axis=-1)([source_image_input, target_domain])
    else:
        source_domain_input = layers.Input(shape=[1], name="source_domain")
        inputs += [source_domain_input]

        source_domain = layers.CategoryEncoding(num_tokens=number_of_domains,
                                                output_mode="one_hot")(source_domain_input)
        source_domain = keras_utils.TileLayer(image_size)(source_domain)
        source_domain = keras_utils.TileLayer(image_size)(source_domain)
        x = layers.Concatenate(axis=-1)([source_image_input, target_domain, source_domain])

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

    return tf.keras.Model(inputs=inputs, outputs=activation, name="StarGANGenerator")


def collagan_affluent_generator(number_of_domains, image_size, output_channels):
    # UnetINDiv4 extracted from:
    # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L941
    def conv_block(block_input, filters, regularizer="l2"):
        # CNR function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L44
        x = block_input
        x = layers.Conv2D(filters, 3, strides=1, padding="same", kernel_regularizer=regularizer)(x)
        x = tfalayers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def downsample(block_input, filters):
        # Pool2d function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L23
        x = layers.Conv2D(filters, 2, strides=2, padding="same", use_bias=False, )(block_input)
        return x

    def upsample__(block_input, filters):
        # Conv2dT function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L29
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(block_input)
        return x

    def conv_1x1__(block_input, filters):
        # Conv1x1 function from (with an additional tanh activation by us):
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py#L38
        x = layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False, activation="tanh")(block_input)
        return x

    source_images_input = layers.Input(shape=[number_of_domains, image_size, image_size, output_channels],
                                       name="source_images")
    target_domain_input = layers.Input(shape=[1], name="target_domain")
    inputs = [source_images_input, target_domain_input]

    target_domain = layers.CategoryEncoding(num_tokens=number_of_domains, output_mode="one_hot")(target_domain_input)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)
    target_domain = keras_utils.TileLayer(image_size)(target_domain)

    # ENCODER starts here...
    base_filters = 64
    filters_per_domain = base_filters // number_of_domains

    source_image_split = tf.unstack(source_images_input, number_of_domains, axis=1)
    affluents_conv_0_2 = []
    affluents_conv_1_2 = []
    affluents_conv_2_2 = []
    affluents_conv_3_2 = []
    affluents_down_4__ = []
    for d in range(number_of_domains):
        conv_0_0 = tf.concat([source_image_split[d], target_domain], axis=-1)
        conv_0_1 = conv_block(conv_0_0, filters_per_domain * 1)
        conv_0_2 = conv_block(conv_0_1, filters_per_domain * 1)
        down_1__ = downsample(conv_0_2, filters_per_domain * 2)
        conv_1_1 = conv_block(down_1__, filters_per_domain * 2)
        conv_1_2 = conv_block(conv_1_1, filters_per_domain * 2)
        down_2__ = downsample(conv_1_2, filters_per_domain * 4)
        conv_2_1 = conv_block(down_2__, filters_per_domain * 4)
        conv_2_2 = conv_block(conv_2_1, filters_per_domain * 4)
        down_3__ = downsample(conv_2_2, filters_per_domain * 8)
        conv_3_1 = conv_block(down_3__, filters_per_domain * 8)
        conv_3_2 = conv_block(conv_3_1, filters_per_domain * 8)
        down_4__ = downsample(conv_3_2, filters_per_domain * 16)

        affluents_conv_0_2 += [conv_0_2]
        affluents_conv_1_2 += [conv_1_2]
        affluents_conv_2_2 += [conv_2_2]
        affluents_conv_3_2 += [conv_3_2]
        affluents_down_4__ += [down_4__]

    # DECODER starts here...
    concat_down_4__ = tf.concat(affluents_down_4__, axis=-1)
    concat_conv_4_1 = conv_block(concat_down_4__, filters_per_domain * 16)
    concat_conv_4_2 = conv_block(concat_conv_4_1, filters_per_domain * 16)
    up_4___________ = upsample__(concat_conv_4_2, filters_per_domain * 8)

    concat_down_3_2 = tf.concat(affluents_conv_3_2, axis=-1)
    concat_skip_3__ = tf.concat([concat_down_3_2, up_4___________], axis=-1)
    up_conv_3_1____ = conv_block(concat_skip_3__, filters_per_domain * 8)
    up_conv_3_2____ = conv_block(up_conv_3_1____, filters_per_domain * 8)
    up_3___________ = upsample__(up_conv_3_2____, filters_per_domain * 4)

    concat_down_2_2 = tf.concat(affluents_conv_2_2, axis=-1)
    concat_skip_2__ = tf.concat([concat_down_2_2, up_3___________], axis=-1)
    up_conv_2_1____ = conv_block(concat_skip_2__, filters_per_domain * 4)
    up_conv_2_2____ = conv_block(up_conv_2_1____, filters_per_domain * 4)
    up_2___________ = upsample__(up_conv_2_2____, filters_per_domain * 2)

    concat_down_1_2 = tf.concat(affluents_conv_1_2, axis=-1)
    concat_skip_1__ = tf.concat([concat_down_1_2, up_2___________], axis=-1)
    up_conv_1_1____ = conv_block(concat_skip_1__, filters_per_domain * 2)
    up_conv_1_2____ = conv_block(up_conv_1_1____, filters_per_domain * 2)
    up_1___________ = upsample__(up_conv_1_2____, filters_per_domain * 1)

    concat_down_0_2 = tf.concat(affluents_conv_0_2, axis=-1)
    concat_skip_0__ = tf.concat([concat_down_0_2, up_1___________], axis=-1)
    up_conv_0_1____ = conv_block(concat_skip_0__, filters_per_domain * 1)
    up_conv_0_2____ = conv_block(up_conv_0_1____, filters_per_domain * 1)

    # added beyond CollaGAN to make pixel values between [-1,1]
    output = conv_1x1__(up_conv_0_2____, output_channels)

    return tf.keras.Model(inputs=inputs, outputs=output, name="CollaGANAffluentGenerator")


def collagan_original_discriminator(number_of_domains, image_size, output_channels):
    # Discriminator adapted from:
    # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/CollaGAN_fExp8.py#L521

    def downsample(block_input, filters):
        # Conv2d2x2 + lReLU function from:
        # https://github.com/jongcye/CollaGAN_CVPR/blob/509cb1dab781ccd4350036968fb3143bba19e1db/model/netUtil.py
        x = block_input
        x = layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=False, )(x)
        x = layers.LeakyReLU()(x)
        return x

    base_filters = 64

    source_image = layers.Input(shape=[image_size, image_size, output_channels], name="source_image")

    conv_0______ = downsample(source_image, base_filters * 1)
    conv_1______ = downsample(conv_0______, base_filters * 2)
    conv_2______ = downsample(conv_1______, base_filters * 4)
    conv_3______ = downsample(conv_2______, base_filters * 8)
    conv_4______ = downsample(conv_3______, base_filters * 16)
    conv_last___ = downsample(conv_4______, base_filters * 32)

    conv_last___ = layers.Dropout(0.5)(conv_last___)

    # outputs: patches + classification
    patches = layers.Conv2D(1, 3, strides=1, padding="same", use_bias=False, name="discriminator_patches")(conv_last___)

    downsampling_blocks = 6
    full_kernel_size = image_size // (2 ** downsampling_blocks)
    classification = layers.Conv2D(number_of_domains, kernel_size=full_kernel_size, strides=1, use_bias=False)(
        conv_last___)
    classification = layers.Reshape((number_of_domains,), name="domain_classification")(classification)

    return tf.keras.Model(inputs=source_image, outputs=[patches, classification], name="CollaGANDiscriminator")
