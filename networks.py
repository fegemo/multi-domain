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


def stargan_resnet_generator(image_size, output_channels, number_of_domains, receive_source_domain, capacity=1):
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

    filters = 64 * capacity
    x = layers.Conv2D(filters, kernel_size=7, strides=1, padding="same", kernel_initializer=init, use_bias=False)(x)
    x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
    x = layers.ReLU()(x)

    # downsampling blocks: 128*cap, then 256*cap
    for i in range(2):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init, use_bias=False)(x)
        x = tfalayers.InstanceNormalization(epsilon=0.00001)(x)
        x = layers.ReLU()(x)

    # bottleneck blocks
    for i in range(6):
        x = resblock(x, filters, 3, init)

    # upsampling blocks: 128*cap, then 64*cap
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


def collagan_affluent_generator(number_of_domains, image_size, output_channels, capacity=1):
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
    base_filters = 64 * capacity
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


def collagan_original_discriminator(number_of_domains, image_size, output_channels, receive_source_image):
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

    real_or_fake_image = layers.Input(shape=[image_size, image_size, output_channels], name="real_or_fake_image")
    inputs = [real_or_fake_image]
    if receive_source_image:
        source_images = layers.Input(shape=[image_size, image_size, output_channels * number_of_domains],
                                     name="source_images")
        inputs += [source_images]

    x___________ = layers.Concatenate(axis=-1)(inputs)

    conv_0______ = downsample(x___________, base_filters * 1)
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

    return tf.keras.Model(inputs=inputs, outputs=[patches, classification], name="CollaGANDiscriminator")


# ----------------------------------------------------------------------------------------------------------------------
# start of MUNIT code
def munit_content_encoder(domain_letter, image_size, channels, number_of_input_images=1):
    """
    From the reference pytorch implementation:
    https://github.com/NVlabs/MUNIT/blob/master/networks.py#L245
    content_encoder:
    - conv   3-> 64  relu reflect instno (7x7 kernel, 1 stride, 3 pad)
    - conv  64->128  relu reflect instno (4x4 kernel, 2 stride, 1 pad)
    - conv 128->256  relu reflect instno (4x4 kernel, 2 stride, 1 pad)
    - 4x resblocks:
        - conv 256->256  relu reflect instno (3x3 kernel, 1 stride, 1 pad)
        - conv 256->256  none reflect instno (3x3 kernel, 1 stride, 1 pad)
        - add  x + 2nd conv
    :param domain_letter: initial representing the domain
    :param image_size: size of the input images
    :param channels: number of channels of the input images
    :param number_of_input_images number of input images, which is 1 for MUNIT but d for ReMIC
    :return: model that encodes the content, outputing a 16x16x256 tensor
    """
    def resblock_content(input_tensor, filters):
        y = input_tensor
        for i in range(2):
            y = keras_utils.ReflectPadding(1)(y)
            y = layers.Conv2D(filters, 3, strides=1, padding="valid", kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(y)
            y = tfalayers.InstanceNormalization(epsilon=1e-5)(y)
            if i == 0:
                y = layers.ReLU()(y)
        y = layers.Add()([y, input_tensor])
        return y

    if number_of_input_images > 1:
        # the single content encoder in ReMIC receives multiple images as input (first dimension0. Then, we permute
        # the images to the last dimension and reshape the tensor to have the images concatenated in the last dimension
        input_layer = layers.Input(shape=(number_of_input_images, image_size, image_size, channels))
        x = layers.Permute((2, 3, 4, 1))(input_layer)
        x = layers.Reshape((image_size, image_size, channels * number_of_input_images))(x)
    else:
        # each content encoder in MUNIT receives a single image as input
        input_layer = layers.Input(shape=(image_size, image_size, channels))
        x = input_layer
    x = keras_utils.ReflectPadding(3)(x)
    x = layers.Conv2D(64, 7, strides=1, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
    x = tfalayers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # 2x downsampling blocks
    x = munit_conv_block(x, 128, 4, 2, True)
    x = munit_conv_block(x, 256, 4, 2, True)

    # 4x residual blocks
    x = resblock_content(x, 256)
    x = resblock_content(x, 256)
    x = resblock_content(x, 256)
    x = resblock_content(x, 256)

    content_code = x
    return tf.keras.Model(input_layer, content_code, name=f"ContentEncoder{domain_letter.upper()}")


def munit_style_encoder(domain_letter, image_size, channels):
    """
    From the reference pytorch implementation:
    https://github.com/NVlabs/MUNIT/blob/master/networks.py#L188
    style_encoder:
    - conv   3-> 64  relu reflect nonorm (7x7 kernel, 1 stride, 3 pad)
    - conv  64->128  relu reflect nonorm (4x4 kernel, 2 stride, 1 pad)
    - conv 128->256  relu reflect nonorm (4x4 kernel, 2 stride, 1 pad)
    - conv 256->256  relu reflect nonorm (4x4 kernel, 2 stride, 1 pad)
    - conv 256->256  relu reflect nonorm (4x4 kernel, 2 stride, 1 pad)
    - globalavgpool 256
    - conv 256->  8  relu    zero nonorm (1x1 kernel, 1 stride, 0 pad)

    :param domain_letter: initial representing the domain
    :param image_size: size of the input images
    :param channels: number of channels of the input images
    :return: model that encodes the style, outputing an 8-dimensional vector
    """
    input_layer = layers.Input(shape=(image_size, image_size, channels))
    x = keras_utils.ReflectPadding(3)(input_layer)
    x = layers.Conv2D(64, 7, strides=1, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu")(x)

    # 4x downscale blocks
    x = munit_conv_block(x, 128, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)

    x = layers.GlobalAvgPool2D(keepdims=True)(x)
    style_code = layers.Conv2D(8, kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    style_code = layers.Reshape((8,))(style_code)
    return tf.keras.Model(input_layer, style_code, name=f"StyleEncoder{domain_letter.upper()}")


def munit_decoder(domain_letter, channels):
    """
    From the reference pytorch implementation:
    https://github.com/NVlabs/MUNIT/blob/master/networks.py#L223
    decoder:
    - 4x resblocks:
        - conv  256->256 relu reflect adainor (3x3 kernel, 1 stride, 1 pad)
        - conv  256->256 none reflect adainor (3x3 kernel, 1 stride, 1 pad)
    - upsa
    - conv  256->128 relu reflect lnnorm (5x5 kernel, 1 stride, 2 pad)
    - upsa
    - conv  128->64 relu reflect lnnorm (5x5 kernel, 1 stride, 2 pad)
    - conv   64->4  tanh reflect nonorm (7x7 kernel, 1 stride, 3 pad)
    :param domain_letter: initial representing the domain
    :param channels: number of channels of the input images
    :return: model that decodes the image, outputing a 64x64x4 tensor
    """
    def mlp_munit():
        input_style_code = layers.Input(shape=(8,))
        adain_params_inner = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu")(
            input_style_code)
        adain_params_inner = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu")(
            adain_params_inner)
        # 4096 = 256 (dense dimension) * 8 (style code) * 2 (weight and bias)
        adain_params_inner = layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(
            adain_params_inner)
        return tf.keras.Model(input_style_code, [adain_params_inner])

    def op_adain(input_tensor):
        y = input_tensor[0]
        mean, var = tf.nn.moments(y, axes=[1, 2], keepdims=True)
        adain_bias = input_tensor[1]
        adain_bias = tf.reshape(adain_bias, [-1, 1, 1, 256])
        adain_scale = input_tensor[2]
        adain_scale = tf.reshape(adain_scale, [-1, 1, 1, 256])
        # tf.print("y", y.shape, "mean", mean.shape, "var", var.shape, "adain_bias", adain_bias.shape, "adain_scale",)
        output_tensor = tf.nn.batch_normalization(y, mean, var, adain_bias, adain_scale, 1e-7)
        # tf.print("output_tensor", output_tensor.shape)
        return output_tensor

    def adaptive_instance_norm2d(input_tensor, adain_params_inner, idx_adain):
        assert input_tensor.shape[-1] == 256
        # tf.print("input_tensor", input_tensor.shape, "adain_params_inner", adain_params_inner.shape, "idx_adain", idx_adain)
        y = input_tensor
        idx_head = idx_adain * 2 * 256
        adain_scale = layers.Lambda(lambda z: z[:, idx_head:idx_head + 256])(adain_params_inner)
        adain_bias = layers.Lambda(lambda z: z[:, idx_head + 256:idx_head + 512])(adain_params_inner)
        output_tensor = layers.Lambda(op_adain)([y, adain_bias, adain_scale])
        # tf.print("output_tensor", output_tensor.shape)
        return output_tensor

    def resblock_adain(input_tensor, filters, adain_params_inner, idx_adain):
        y = input_tensor
        y = keras_utils.ReflectPadding(1)(y)
        y = layers.Conv2D(filters, 3, strides=1, padding="valid", kernel_initializer="he_normal",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          bias_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(y)
        y = layers.Lambda(lambda z: adaptive_instance_norm2d(z[0], z[1], idx_adain))([y, adain_params_inner])
        y = layers.ReLU()(y)
        y = keras_utils.ReflectPadding(1)(y)
        y = layers.Conv2D(filters, 3, strides=1, padding="valid", kernel_initializer="he_normal",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          bias_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(y)
        y = layers.Lambda(lambda z: adaptive_instance_norm2d(z[0], z[1], idx_adain + 1))([y, adain_params_inner])
        y = layers.Add()([y, input_tensor])
        return y

    input_style = layers.Input(shape=(8,))
    style_code = input_style
    mlp = mlp_munit()
    adain_params = mlp(style_code)

    input_content = layers.Input(shape=(16, 16, 256))
    w = content_code = input_content

    # 4x resblocks
    w = resblock_adain(w, 256, adain_params, 0)
    w = resblock_adain(w, 256, adain_params, 2)
    w = resblock_adain(w, 256, adain_params, 4)
    w = resblock_adain(w, 256, adain_params, 6)

    # 2x upscale blocks
    w = munit_upscale_nn(w, 128)
    w = munit_upscale_nn(w, 64)
    x = keras_utils.ReflectPadding(3)(w)
    output_image = layers.Conv2D(channels, 7, strides=1, padding="valid",
                                 kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                 activation="tanh")(x)
    return tf.keras.Model([input_style, input_content], [output_image, style_code, content_code],
                          name=f"Decoder{domain_letter.upper()}")


def munit_conv_block(input_tensor, filters, kernel_size=3, strides=2, use_norm=False):
    x = input_tensor
    x = keras_utils.ReflectPadding(1)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=(not use_norm))(x)
    if use_norm:
        x = tfalayers.InstanceNormalization(epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    return x


def munit_conv_block_d(input_tensor, filters, use_norm=False):
    x = input_tensor
    x = keras_utils.ReflectPadding(2)(x)
    x = layers.Conv2D(filters, 4, strides=2, padding="valid",
                      kernel_initializer=tf.random_normal_initializer(0, 0.02),
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=(not use_norm))(x)
    if use_norm:
        x = tfalayers.InstanceNormalization(epsilon=1e-5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def munit_upscale_nn(input_tensor, filters, use_norm=False):
    x = input_tensor
    x = layers.UpSampling2D()(x)
    x = keras_utils.ReflectPadding(2)(x)
    x = layers.Conv2D(filters, 5, strides=1, padding="valid",
                      kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      use_bias=(not use_norm))(x)
    if use_norm:
        x = layers.GroupNormalization(group=filters)(x)
    x = layers.ReLU()(x)
    return x


def munit_discriminator_multi_scale(domain_letter, image_size, channels, scales):
    """
    From pytorch reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py#L35:
    discriminator_ms (3 redes idênticas a esta:)
    - conv   3-> 64 lrelu reflect nonorm (4x4 kernel, 2 stride)
    - conv  64->128 lrelu reflect nonorm
    - conv 128->256 lrelu reflect nonorm
    - conv 256->512 lrelu reflect nonorm
    - conv 512->  1  relu    zero nonorm (1x1 kernel, 1 stride)

    executa as redes com 0 avgpools, 1x avgpools e 2x avgpools (i.e., 64x64, 32x32, 16x16)
    - avgp        (3x3 kernel, 2 stride) -> pra ficar com 1 só valor pra imagem toda

    passa a imagem com original 256 (original da munit)
    passa também avgpooled com  128
    passa também avgpooled com   64
    :param channels: number of channels of the image
    :param image_size: size of the input images
    :param domain_letter: domain representation ('B', 'L', 'F', 'R')
    :return: the discriminator keras model
    """
    def conv2d_blocks(input_tensor):
        x = input_tensor
        x = munit_conv_block_d(x, 64)
        x = munit_conv_block_d(x, 128)
        x = munit_conv_block_d(x, 256)
        x = munit_conv_block_d(x, 512)
        x = layers.Conv2D(1, kernel_size=1, kernel_initializer="he_normal",
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=True, padding="valid")(x)
        return x

    input_layer = layers.Input(shape=(image_size, image_size, channels))
    x = input_layer
    outputs = []
    for i in range(scales):
        outputs.append(conv2d_blocks(x))
        x = layers.AveragePooling2D(pool_size=(3, 3), strides=2)(x)

    if scales == 1:
        shape = list(map(lambda d: d.value, outputs[0].shape.dims[1:]))
        outputs[0] = layers.Reshape((1, *shape))(outputs[0])
    return tf.keras.Model([input_layer], outputs, name=f"Discriminator{domain_letter.upper()}")


def remic_unified_content_encoder(domain_letter, image_size, channels, number_of_input_images):
    return munit_content_encoder(domain_letter, image_size, channels, number_of_input_images)


def remic_style_encoder(domain_letter, image_size, channels):
    def resblock_style(input_tensor, filters):
        """
        We just copied this function from the MUNIT implementation, as the ReMIC paper does not describe the style
        encoder resblock structure (neither the number of resblocks).
        We actually removed the instance normalization, as the paper says it is not used in the style encoder
        :param input_tensor:
        :param filters:
        :return:
        """
        y = input_tensor
        for i in range(2):
            y = keras_utils.ReflectPadding(1)(y)
            y = layers.Conv2D(filters, 3, strides=1, padding="valid", kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(y)
            if i == 0:
                y = layers.ReLU()(y)
        y = layers.Add()([y, input_tensor])
        return y

    input_layer = layers.Input(shape=(image_size, image_size, channels))
    x = keras_utils.ReflectPadding(3)(input_layer)
    x = layers.Conv2D(64, 7, strides=1, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), activation="relu")(x)

    # 4x downscale blocks
    x = munit_conv_block(x, 128, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)
    x = munit_conv_block(x, 256, kernel_size=4)

    # the paper does not describe how many resblocks (it says "several"), so we suppose it is 4
    # (as in the content encoder)
    for _ in range(4):
        x = resblock_style(x, 256)

    x = layers.GlobalAvgPool2D(keepdims=True)(x)
    style_code = layers.Conv2D(8, kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    style_code = layers.Reshape((8,))(style_code)
    return tf.keras.Model(input_layer, style_code, name=f"StyleEncoder{domain_letter.upper()}")


def remic_generator(domain_letter, channels):
    # def adain_resblock(x, style, filters, kernel_size, adain_filters, init="he_normal"):
    #     original_x = x
    #
    #     b = layers.Dense(adain_filters)(style)
    #     b = layers.Reshape([1, 1, adain_filters])(b)
    #     g = layers.Dense(adain_filters)(style)
    #     g = layers.Reshape([1, 1, adain_filters])(g)
    #
    #     x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    #     x = keras_utils.AdaInstanceNormalization()([x, b, g])
    #     x = layers.LeakyReLU()(x)
    #
    #     b = layers.Dense(adain_filters)(style)
    #     b = layers.Reshape([1, 1, adain_filters])(b)
    #     g = layers.Dense(adain_filters)(style)
    #     g = layers.Reshape([1, 1, adain_filters])(g)
    #
    #     x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init, use_bias=False)(x)
    #     x = keras_utils.AdaInstanceNormalization()([x, b, g])
    #     x = layers.Add()([original_x, x])
    #
    #     return x
    #
    # latent_input_layer = layers.Input(shape=(16, 16, 4))
    # style_input_layer = layers.Input(shape=(8,))
    #
    # x = latent_input_layer
    # s = style_input_layer
    #
    # # 4x residual blocks
    # for _ in range(4):
    #     x = adain_resblock(x, s, 256, 3, style_code_length)
    #
    # # 2x upsample blocks
    # x = layers.UpSampling2D()(x)
    # x = layers.Conv2D(128, kernel_size=5, strides=1, padding="same")(x)
    # x = layers.UpSampling2D()(x)
    # x = layers.Conv2D(64, kernel_size=5, strides=1, padding="same")(x)
    #
    # x = layers.Conv2D(4, kernel_size=7, strides=1, padding="same", activation="tanh")(x)
    # output_layer = x
    #
    # return tf.keras.Model([latent_input_layer, style_input_layer], [output_layer], name="Generator")
    return munit_decoder(domain_letter, channels)


def remic_discriminator(domain_letter, image_size, channels, scales):
    # input_layer = layers.Input(shape=(64, 64, 4))
    # x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(input_layer)
    # x = layers.LeakyReLU(0.2)(x)
    # x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    # x = layers.LeakyReLU(0.2)(x)
    # x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    # x = layers.LeakyReLU(0.2)(x)
    # x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
    # x = layers.LeakyReLU(0.2)(x)
    return munit_discriminator_multi_scale(domain_letter, image_size, channels, scales)
