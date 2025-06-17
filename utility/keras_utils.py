from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow import keras, RaggedTensorSpec
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


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
        t = tf.cast(tf.divide(step, self.total_steps), tf.float32)
        down_slope_value = tf.cast(self.initial_learning_rate * (-t + 1.) * 2., tf.float32)
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


# ------ layer in which each palette can have a different number of colors ------
# this refrains from using vectorized operations, which make it slower for longer batches
class DifferentiablePaletteQuantization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = self.add_weight(shape=(), name="temperature", trainable=False)

    def call(self, inputs, training=None):
        """
        Returns the image with its colors quantized to the palette. During training, it is done with
        a soft assignment using the softmax function with a temperature. During inference, it is done
        with a hard assignment, using the closest color in the palette (hence, losing differentiability).

        :param inputs: Tuple of (images, palettes), with shapes [b, h, w, c] and [b, (k), c] respectively
        :param training: True if training (uses soft assignment through softmax with temperature)
            or False otherwise (uses hard assignment, losing differentiability)
        :return: Quantized images: Tensor of shape [b, h, w, c]
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

            if training:
                weights = tf.nn.softmax(-distances / self.temperature, axis=-1)
                return tf.einsum('...k,kc->...c', weights, palette)  # [H, W, C]
            else:
                indices = tf.argmin(distances, axis=-1)
                return tf.gather(palette, indices)

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


class GumbelSoftmaxPaletteQuantization(Layer):
    def __init__(self, max_palette_size, **kwargs):
        super().__init__(**kwargs)
        self.temperature = self.add_weight(shape=(), name="temperature", trainable=False)
        self.max_palette_size = max_palette_size
        self.invalid_color = tf.constant([-1., -1., -1., -1.], dtype=tf.float32)

    def call(self, inputs, training=None):
        """
        Returns the image with its colors quantized to the palette. During training, it is done with
        a soft assignment using the softmax function with a temperature. During inference, it is done
        with a hard assignment, using the palette clor with the closest logit (hence, losing differentiability).

        :param inputs: Tuple of (images, palettes), with shapes [b, h, w, c] and [b, (k), c] respectively
        :param training: True if training (uses soft assignment through softmax with temperature)
            or False otherwise (uses hard assignment, losing differentiability)
        :return: Quantized images: Tensor of shape [b, h, w, c]
        """
        images, palettes = inputs
        channels = palettes.shape[-1]

        # Process each (image, palette) pair independently
        def quantize_single_image(args):
            # img: [s, s, c]
            # palette: [nc, c] (variable nc)
            img, palette = args
            if training:
                eps = 1e-20
                uniform_noise = tf.random.uniform(tf.shape(img), 0, 1)
                gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + eps) + eps)
                palette_probs = tf.nn.softmax((img + gumbel_noise) / self.temperature)
            else:
                palette_probs = tf.one_hot(tf.argmax(img, axis=-1), self.max_palette_size)
            palette_expanded = palette[tf.newaxis, tf.newaxis, ...]
            # palette_expanded (shape=[1, 1, nc, c])
            palette_probs_expanded = palette_probs[..., tf.newaxis]
            # palette_probs_expanded (shape=[s, s, nc, 1])
            quantized_image = tf.reduce_sum(palette_probs_expanded * palette_expanded, axis=-2)
            # quantized_image (shape=[s, s, c])
            return quantized_image

        # there is no easy way to vectorize this operation, so we use map_fn to
        # process each <image,palette> pair independently
        images_shape = images.shape
        image_size = images_shape[1]
        # tf.print("palettes.shape", palettes.shape)
        # tf.print("tf.shape(palettes)", tf.shape(palettes))
        if isinstance(palettes, tf.RaggedTensor):
            palettes = palettes.to_tensor(default_value=self.invalid_color, shape=[images_shape[0], self.max_palette_size, channels])
        elif palettes.shape[1] < self.max_palette_size:
            number_of_colors = palettes.shape[1]
            palettes = tf.pad(palettes, [[0, 0], [0, self.max_palette_size - number_of_colors], [0, 0]], constant_values=-1.)
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


class PaletteTransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads

    def build(self, input_shape):
        self.proj = layers.Dense(self.embed_dim)
        self.enc_layers = [
            layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.head_dim
            ) for _ in range(self.num_layers)
        ]
        self.layer_norms = [layers.LayerNormalization() for _ in range(self.num_layers)]

    def call(self, inputs):
        # print(f"inputs.shape: {inputs.shape}")
        # tf.print(f"inputs.shape: {inputs.shape}")
        if isinstance(inputs, tf.RaggedTensor):
            colors = inputs.to_tensor()  # [batch, max_colors, channels]
            mask = tf.sequence_mask(inputs.row_lengths(), tf.shape(colors)[1])
        else:
            colors = inputs
            mask = tf.ones((tf.shape(colors)[0], tf.shape(colors)[1]), dtype=tf.bool)

        num_colors = tf.shape(colors)[1]

        # Project to embedding space
        x = self.proj(colors)  # [batch, num_colors, embed_dim]

        # Prepare 3D attention mask [batch, num_heads, num_colors]
        attention_mask = tf.repeat(mask[:, tf.newaxis, ...], repeats=num_colors, axis=1)
        # [batch, num_heads, num_colors]

        # Transformer layers
        for attn, ln in zip(self.enc_layers, self.layer_norms):
            # MultiHeadAttention expects [batch, seq_len, embed_dim]
            x_attn = attn(x, x, x, attention_mask=attention_mask)
            x = ln(x + x_attn)

        # Masked mean pooling
        mask_expanded = tf.expand_dims(tf.cast(mask, x.dtype), -1)  # [batch, num_colors, 1]
        # tf.print(f"mask_expanded.shape: {mask_expanded.shape}")
        masked_count = tf.reduce_sum(mask_expanded, axis=1)
        # tf.print(f"masked_count.shape: {masked_count.shape}")
        masked_x = tf.reduce_sum(x * mask_expanded, axis=1)  # [batch, embed_dim]
        # tf.print(f"masked_x.shape: {masked_x.shape}")
        return masked_x / masked_count

    def compute_output_spec(self, inputs):
        return keras.KerasTensor(shape=(inputs.shape[0], self.embed_dim), dtype="float32")


class PaletteConditioner(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def build(self, input_shapes):
        # input_shapes: [(batch, H, W, C), (batch, embed_dim)]
        image_shape, palette_shape = input_shapes

        # Projections
        self.query_proj = layers.Dense(self.embed_dim)
        self.key_proj = layers.Dense(self.embed_dim)
        self.value_proj = layers.Dense(self.embed_dim)

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim
        )
        self.output_proj = layers.Dense(image_shape[-1])  # Match input channels
        self.built = True

    def call(self, inputs):
        image_features, palette_embed = inputs

        # Project queries (image features)
        queries = self.query_proj(image_features)  # [B, H, W, embed_dim]
        batch_size = tf.shape(queries)[0]
        h, w = tf.shape(queries)[1], tf.shape(queries)[2]

        # Reshape queries to [B, seq_len, num_heads, head_dim]
        queries_flat = tf.reshape(queries, [batch_size, h * w, self.embed_dim])
        queries_4d = tf.reshape(
            queries_flat,
            [batch_size, h * w, self.num_heads, self.head_dim]
        )

        # Project keys/values (palette)
        keys = self.key_proj(palette_embed)  # [B, embed_dim]
        values = self.value_proj(palette_embed)

        # Expand palette to match attention dims
        keys = tf.reshape(
            keys,
            [batch_size, 1, self.num_heads, self.head_dim]
        )
        values = tf.reshape(
            values,
            [batch_size, 1, self.num_heads, self.head_dim]
        )

        # Cross-attention
        attended = self.attention(
            query=queries_4d,
            key=keys,
            value=values,
            return_attention_scores=False
        )  # [B, h*w, num_heads, head_dim]

        # Reshape back to spatial
        attended = tf.reshape(
            attended,
            [batch_size, h, w, self.embed_dim]
        )
        return self.output_proj(attended)


class AnnealingScheduler(ABC):
    def __init__(self, annealing_layers=None):
        if annealing_layers is None:
            annealing_layers = []
        self.annealing_layers = annealing_layers

    def update(self, t):
        new_temperature = self.get_value(t)
        for l in self.annealing_layers:
            l.temperature.assign(new_temperature)
        return new_temperature

    @abstractmethod
    def get_value(self, t):
        pass


class LinearAnnealingScheduler(AnnealingScheduler):
    def __init__(self, initial_temperature, layers):
        super().__init__(layers)
        self.initial_temperature = initial_temperature

    def get_value(self, t):
        return tf.maximum(0.0, (1.0 - t) * self.initial_temperature)


class NoopAnnealingScheduler(AnnealingScheduler):
    def get_value(self, t):
        return 1.0
