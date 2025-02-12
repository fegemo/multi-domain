import tensorflow as tf
from tensorflow import keras, RaggedTensorSpec
from tensorflow.keras.layers import Layer

class ConstantThenLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        return super().get_config()

    # ____
    #     \
    #      \
    def __init__(self, initial_learning_rate, total_steps):
        self.initial_learning_rate = initial_learning_rate
        self.total_steps = tf.cast(total_steps, "float32")

    def __call__(self, step):
        t = tf.divide(step, self.total_steps)
        down_slope_value = self.initial_learning_rate * (-t + 1.) * 2.
        return tf.maximum(0.0, tf.minimum(self.initial_learning_rate, down_slope_value))


class TileLayer(keras.layers.Layer):
    def __init__(self, number):
        super(TileLayer, self).__init__()
        self.multiples = None
        self.number = number

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        self.multiples = tf.TensorShape([1, self.number] + ([1] * (len(input_shape) - 1)))
        super(TileLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        higher_rank_inputs = tf.expand_dims(inputs, 1)
        return tf.tile(higher_rank_inputs, self.multiples, name="TileLayer")

    def get_config(self):
        return {"number": self.number}

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], self.number] + input_shape[1:])


class NParamsSupplier:
    def __init__(self, supply_first_n_params):
        self.n = supply_first_n_params

    def __call__(self, *args, **kwargs):
        return [*args[:self.n]]


class ReflectPadding(keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        super(ReflectPadding, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0]],
                      mode="REFLECT")

    def get_config(self):
        config = super(ReflectPadding, self).get_config()
        config.update({"padding": self.padding})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.padding * 2, input_shape[2] + self.padding * 2, input_shape[3]


# Got from: https://github.com/manicman1999/StyleGAN-Keras/blob/master/AdaIN.py
# Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Calculates the AdaIN normalization of the inputs [x, beta, gamma]
        :param inputs: list of [x, beta, gamma] tensors
        :param training: unused
        :return:
        """
        x, beta, gamma = inputs[0], inputs[1], inputs[2]

        input_shape = tf.shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean, var = tf.nn.moments(inputs[0], axes=reduction_axes, keepdims=True)
        stddev = tf.sqrt(var) + self.epsilon
        normalized_x = (inputs[0] - mean) / stddev

        return normalized_x * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def count_network_parameters(network):
    return tf.reduce_sum([tf.reduce_prod(v.shape) for v in network.trainable_weights])


# class DifferentiablePalettePerImage(Layer):
#     def __init__(self, temperature=1.0, **kwargs):
#         super().__init__(**kwargs)
#         self.temperature = temperature
#
#     def call(self, inputs):
#         """
#         Args:
#             inputs: Tuple of (images, palettes)
#             - images: Tensor of shape [B, H, W, C] (channels_last)
#             - palettes: Tensor of shape [B, K, C] (different K per batch allowed)
#         Returns:
#             Quantized images: Tensor of shape [B, H, W, C]
#         """
#         images, palettes = inputs
#
#         # Add extra dimensions for broadcasting
#         images_expanded = tf.expand_dims(images, axis=-2)  # [B, H, W, 1, C]
#         palettes_expanded = tf.expand_dims(palettes, axis=1)  # [B, 1, 1, K, C]
#         palettes_expanded = tf.expand_dims(palettes_expanded, axis=1)
#
#         # Compute squared distances between pixels and palette colors
#         distances = tf.reduce_sum(
#             (images_expanded - palettes_expanded) ** 2,
#             axis=-1
#         )  # [B, H, W, K]
#
#         # Convert distances to weights using softmax with temperature
#         weights = tf.nn.softmax(-distances / self.temperature, axis=-1)
#
#         # Weighted sum of palette colors
#         quantized = tf.einsum("bhwk,bkc->bhwc", weights, palettes)
#
#         return quantized
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({"temperature": self.temperature})
#         return config
#
#
# # Usage example:
# if __name__ == "__main__":
#     # Create layer
#     palette_layer = DifferentiablePalettePerImage(temperature=1.0)
#
#     # Create dummy input (2 images in batch)
#     batch_size = 2
#     img_size = 1
#     channels = 4
#
#     # Random images [B, H, W, C]
#     generated_images = tf.random.uniform((batch_size, img_size, img_size, channels))
#
#     # Different palette per image [B, K, C]
#     palettes = tf.ragged.constant([
#         # First image's palette (3 colors)
#         [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],# [0.5, 0.5, 0.5, 1.0]],
#         # Second image's palette (3 different colors)
#         [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
#     ], dtype=tf.float32)
#
#     # Process images
#     quantized_images = palette_layer((generated_images, palettes))
#
#     print("Input images shape:", generated_images.shape)
#     print("Palettes shape:", palettes.shape)
#     print("Output shape:", quantized_images.shape)
#     print("Output 1 max:", tf.reduce_max(quantized_images[0]).numpy())
#     print("Output 2 max:", tf.reduce_max(quantized_images[1]).numpy())
#     print("generated_images[0]:", generated_images[0])
#     print("quantized[0]:", quantized_images[0])
#     print("generated_images[1]:", generated_images[1])
#     print("quantized[1]:", quantized_images[1])

# ------ layer in which each palette can have a different number of colors ------
# this refrains from using vectorized operations, which make it slower for longer batches
class DynamicDifferentiablePalette(Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = tf.Variable(temperature, trainable=False)

    def call(self, inputs):
        """
        Args:
            inputs: Tuple of (images, palettes)
            - images: Tensor of shape [B, H, W, C] (channels_last)
            - palettes: RaggedTensor of shape [B, (K), C] (variable K per batch)
        Returns:
            Quantized images: Tensor of shape [B, H, W, C]
        """
        images, palettes = inputs

        # Process each (image, palette) pair independently
        def quantize_single_image(args):
            # img: [H, W, C]
            # palette: [K, C] (variable K)
            img, palette = args
            distances = tf.reduce_sum(
                (tf.expand_dims(img, -2) - tf.expand_dims(palette, 0)) ** 2,
                axis=-1
            )  # [H, W, K]

            weights = tf.nn.softmax(-distances / self.temperature, axis=-1)
            return tf.einsum('...k,kc->...c', weights, palette)  # [H, W, C]

        # there is no easy way to vectorize this operation, so we use map_fn to
        # process each <image,palette> pair independently
        images_shape = images.shape
        image_size, channels = images_shape[1], images_shape[3]
        return tf.map_fn(
            fn=quantize_single_image,
            elems=(images, palettes),
            fn_output_signature=tf.TensorSpec([image_size, image_size, channels], tf.float32),
            infer_shape=False
        )

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# # Usage example:
# if __name__ == "__main__":
#     # Create layer
#     palette_layer = DynamicDifferentiablePalette(temperature=0.01)
#
#     # Batch of 3 images with different palette sizes
#     batch_size = 3
#     img_size = 2
#     channels = 3
#
#     # Random images [B, H, W, C]
#     generated_images = tf.random.uniform((batch_size, img_size, img_size, channels))
#
#     # Ragged palettes (different K per image)
#     palettes = tf.ragged.constant([
#         # Image 1: 2 colors
#         [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
#         # Image 2: 3 colors
#         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
#         # Image 3: 4 colors
#         [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.5, 0.5]]
#     ], dtype=tf.float32)
#
#     # Process images
#     quantized_images = palette_layer((generated_images, palettes))
#
#     print("Input shape:", generated_images.shape)
#     print("Palettes:", palettes)
#     print("Output shape:", quantized_images.shape)
#     print("Output 0 max:", tf.reduce_max(quantized_images[0]).numpy())
#     print("Output 1 max:", tf.reduce_max(quantized_images[1]).numpy())
#     print("Output 2 max:", tf.reduce_max(quantized_images[2]).numpy())
#     print("generated_images[0]:", generated_images[1])
#     print("quantized_images[0]:", quantized_images[1])


# ---- layer that quantizes to same-size palettes that are padded with an invalid value ----
# it uses a mask to indicate which colors are valid
# uses vectorized operations, but might use much more memory than necessary
# class MaskedDifferentiablePalette(Layer):
#     def __init__(self, max_colors=64, temperature=1.0, invalid_value=-1.0, **kwargs):
#         super().__init__(**kwargs)
#         self.max_colors = max_colors
#         self.temperature = temperature
#         self.invalid_value = invalid_value
#
#     def call(self, inputs):
#         """
#         Args:
#             inputs: Tuple of (images, palettes)
#             - images: Tensor of shape [B, H, W, C]
#             - palettes: Tensor of shape [B, max_colors, C] (padded with invalid_value)
#         Returns:
#             Quantized images: Tensor of shape [B, H, W, C]
#         """
#         images, palettes = inputs
#
#         # Create validity mask [B, max_colors]
#         validity_mask = tf.reduce_any(
#             tf.not_equal(palettes, self.invalid_value),
#             axis=-1
#         )  # True for valid colors
#
#         # Replace invalid colors with zeros (won't affect distance calculation)
#         sanitized_palettes = tf.where(
#             tf.expand_dims(validity_mask, -1),
#             palettes,
#             tf.zeros_like(palettes)
#         )
#
#         # Calculate distances [B, H, W, max_colors]
#         images_exp = tf.expand_dims(images, 3)  # [B, H, W, 1, C]
#         palettes_exp = tf.expand_dims(sanitized_palettes, [1, 2])  # [B, 1, 1, K, C]
#         distances = tf.reduce_sum((images_exp - palettes_exp) ** 2, axis=-1)
#
#         # Apply large distance to invalid colors
#         large_distance = 1e9
#         adjusted_distances = tf.where(
#             tf.expand_dims(validity_mask, [1, 2]),  # [B, 1, 1, K]
#             distances,
#             large_distance * tf.ones_like(distances)
#         )
#
#         # Compute weights [B, H, W, max_colors]
#         weights = tf.nn.softmax(-adjusted_distances / self.temperature, axis=-1)
#
#         # Zero out weights for invalid colors
#         masked_weights = weights * tf.cast(tf.expand_dims(validity_mask, [1, 2]), tf.float32)
#
#         # Normalize weights (handle cases with some invalid colors)
#         sum_weights = tf.reduce_sum(masked_weights, axis=-1, keepdims=True) + 1e-8
#         normalized_weights = masked_weights / sum_weights
#
#         # Compute quantized values
#         quantized = tf.einsum('bhwk,bkc->bhwc', normalized_weights, palettes)
#
#         return quantized
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "max_colors": self.max_colors,
#             "temperature": self.temperature,
#             "invalid_value": self.invalid_value
#         })
#         return config
#
#
# # Usage example:
# if __name__ == "__main__":
#     B, H, W, C = 2, 32, 32, 3
#     max_colors = 5
#     invalid = -1.0
#
#     # Example palettes (batch_size=2, max_colors=5, channels=3)
#     palettes = tf.constant([
#         # First image palette: 3 valid colors + 2 invalid
#         [[1, 0, 0], [0, 1, 0], [0, 0, 1], [invalid] * 3, [invalid] * 3],
#         # Second image palette: 2 valid colors + 3 invalid
#         [[1, 1, 0], [0, 1, 1], [invalid] * 3, [invalid] * 3, [invalid] * 3]
#     ], dtype=tf.float32)
#
#     images = tf.random.uniform((B, H, W, C))
#     layer = MaskedDifferentiablePalette(max_colors=5, temperature=0.1)
#     quantized = layer((images, palettes))
#
#     print("Input shape:", images.shape)
#     print("Quantized shape:", quantized.shape)
#     print("First image max values:", tf.reduce_max(quantized[0], axis=[0, 1]))
#     print("Second image max values:", tf.reduce_max(quantized[1], axis=[0, 1]))


class PaletteExtractor(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Extracts the palette of each image in the batch, returning a ragged tensor of shape [b, (colors), c]
        :param inputs: batch of images with shape [B, H, W, C] as tf.float32 in the range [-1, 1]
        :return: batch of palettes as a ragged tensor of shape [B, (colors), C] as tf.float32 in the range [-1, 1]
        """
        images = inputs
        denormalized = tf.cast((images + 1.0) * 127.5, tf.int32)

        palettes = tf.map_fn(
            fn=PaletteExtractor.extract_palette,
            elems=denormalized,
            fn_output_signature=RaggedTensorSpec(
                (None, 4),
                ragged_rank=0,
                dtype=tf.int32)
        )
        palettes = tf.cast(palettes, tf.float32)
        palettes = (palettes / 127.5) - 1.0

        return palettes

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], None, 4])


    @staticmethod
    def extract_palette(image):
        """
        Extracts the palette of a single image
        :param image: a 3d tensor with shape [H, W, C] as tf.int32 in the [0, 255] range
        :return: the image's unique colors in a ragged tensor of shape [colors, C] as tf.int32 in the [0, 255] range
        """
        # reshape to put all pixels of an image in the same dimension
        flat_image = tf.reshape(image, [-1, image.shape[-1]])

        # return the unique values
        colors, _ = tf.raw_ops.UniqueV2(x=flat_image, axis=[0])
        return colors

    def get_config(self):
        config = super().get_config()
        return config