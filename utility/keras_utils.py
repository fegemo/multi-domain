from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow import keras, RaggedTensorSpec
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

from utility.functional_utils import listify


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
    def __init__(self, supply_first_n_params, delistify=False):
        self.n = supply_first_n_params
        self.select_params = lambda args: args[0] if supply_first_n_params == 1 and delistify else args[0:supply_first_n_params]

    def __call__(self, *args, **kwargs):
        return self.select_params(args)


class SkipParamsSupplier:
    def __init__(self, skip_params_with_index=None):
        if skip_params_with_index is None:
            skip_params_with_index = []
        self.indices_to_skip = skip_params_with_index

    def __call__(self, *args, **kwargs):
        params = [args[i] for i in range(len(args)) if i not in self.indices_to_skip]
        return params


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
    def __init__(self, initial_temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = self.add_weight(shape=(), name="temperature", trainable=False,
                                           initializer=tf.constant_initializer(initial_temperature))

    def call(self, inputs, training=None):
        """
        Returns the image with its colors quantized to the palette. During training, it is done with
        a soft assignment using the softmax function with a temperature. During inference, it is done
        with a hard assignment, using the closest color in the palette (hence, losing differentiability).

        :param inputs: Tuple of (images, palettes), with shapes:
            - Option 1: [b, h, w, c] and [b, (k), c] (original format)
            - Option 2: [b, num_domains, h, w, c] and [b, k, c] (format with full example, but does not support
                ragged palettes)
        :param training: True if training (uses soft assignment through softmax with temperature)
            or False otherwise (uses hard assignment, losing differentiability)
        :return: Quantized images: Tensor of shape matching input images shape
        """
        images, palettes = inputs

        # Determine input format based on rank
        if images.shape.rank == 4:
            # single image per item in the batch format: [b, h, w, c]
            return self.quantize_images(images, palettes, training)
        elif images.shape.rank == 5:
            # full example per item in the batch format: [b, num_domains, h, w, c]
            batch_size, num_domains, image_size, channels = (tf.shape(images)[0], tf.shape(images)[1],
                                                             tf.shape(images)[2], tf.shape(images)[-1])

            # Repeat the palette for each domain
            palettes_expanded = tf.tile(tf.expand_dims(palettes, 1), [1, num_domains, 1, 1])
            # palettes_expanded (shape=[b, num_domains, k, c])

            # Reshape to combine batch and domains dimensions
            images_reshaped = tf.reshape(images, [batch_size * num_domains,
                                                  image_size, image_size, channels])
            palettes_reshaped = tf.reshape(palettes_expanded, [batch_size * num_domains, -1, channels])
            # images_reshaped (shape=[b * num_domains, h, w, c])
            # palettes_reshaped (shape=[b * num_domains, k, c])

            # Quantize all images
            quantized = self.quantize_images(images_reshaped, palettes_reshaped, training)

            # Reshape back to original format
            return tf.reshape(quantized, [batch_size, num_domains] + quantized.shape[1:].as_list())
        else:
            raise ValueError(f"Unsupported input rank: {images.shape.rank}. Expected 4 or 5.")

    def quantize_images(self, images, palettes, training):
        """Helper function to quantize images with the given palettes."""

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
                temperature = tf.maximum(self.temperature, 1e-8)
                weights = tf.nn.softmax(-distances / temperature, axis=-1)
                return tf.einsum('...k,kc->...c', weights, palette)  # [H, W, C]
            else:
                indices = tf.argmin(distances, axis=-1)
                return tf.gather(palette, indices)

        # there is no easy way to vectorize this operation, so we use map_fn to
        # process each <image,palette> pair independently
        images_shape = images.shape
        image_size, channels = images_shape[-3], images_shape[-1]
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
        if isinstance(palettes, tf.RaggedTensor):
            palettes = palettes.to_tensor(default_value=self.invalid_color,
                                          shape=[images_shape[0], self.max_palette_size, channels])
        elif palettes.shape[1] < self.max_palette_size:
            number_of_colors = palettes.shape[1]
            palettes = tf.pad(palettes, [[0, 0], [0, self.max_palette_size - number_of_colors], [0, 0]],
                              constant_values=-1.)
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
        masked_count = tf.reduce_sum(mask_expanded, axis=1)
        masked_x = tf.reduce_sum(x * mask_expanded, axis=1)  # [batch, embed_dim]
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
    def __init__(self, initial_temperature, annealing_layers):
        super().__init__(annealing_layers)
        self.initial_temperature = initial_temperature

    def get_value(self, t):
        return tf.maximum(0.0, (1.0 - t) * self.initial_temperature)



class CosineAnnealingScheduler(AnnealingScheduler):
    """
    Cosine annealing scheduler for temperature annealing. It reduces the temperature from the initial value to 0
    following a cosine curve with an amplitude that decreases as it approaches zero.
    Formula: https://www.google.com.br/search?q=f%28x%29%3D%281-x%29%2810%2B3+cos+%28x%E2%8B%8537.7%29%29&sca_esv=37dc570b99b93daa&hl=pt-BR&sxsrf=AE3TifO9WPcA8DRjJyVbKQiHxldxLgaPxQ%3A1755614259740&ei=M4ykaIfoLNyG1sQP87G1kQs&ved=0ahUKEwjHzcHijJePAxVcg5UCHfNYLbIQ4dUDCBA&uact=5&oq=f%28x%29%3D%281-x%29%2810%2B3+cos+%28x%E2%8B%8537.7%29%29&gs_lp=Egxnd3Mtd2l6LXNlcnAiH2YoeCk9KDEteCkoMTArMyBjb3MgKHjii4UzNy43KSkyBRAAGO8FMggQABiABBiiBDIFEAAY7wUyBRAAGO8FSMgyUOESWNsocAJ4AZABAJgBfqAB5wOqAQMwLjS4AQPIAQD4AQGYAgagAvsDwgIKEAAYsAMY1gQYR5gDAIgGAZAGCJIHAzIuNKAH0gmyBwMwLjS4B_EDwgcFMC40LjLIBxA&sclient=gws-wiz-serp
    """
    def __init__(self, initial_temperature, cycles, annealing_layers):
        super().__init__(annealing_layers)
        self.initial_temperature = initial_temperature
        self.cycles = cycles
        self.amplitude_variance = 0.15
        self.FULL_CYCLE = 2 * 3.14159

    def get_value(self, t):
        t0 = self.initial_temperature
        sigma = self.amplitude_variance
        lamb = self.cycles
        CYC = self.FULL_CYCLE
        return (1 - t) * ((1 - sigma) * t0 + (t0 * sigma) * tf.math.cos(t * lamb * CYC))

class ExpCosineAnnealingSchedule(CosineAnnealingScheduler):
    def __init__(self, initial_temperature, cycles, annealing_layers):
        super().__init__(initial_temperature, cycles, annealing_layers)

    def get_value(self, t):
        t0 = self.initial_temperature
        sigma = self.amplitude_variance
        lamb = self.cycles
        CYC = self.FULL_CYCLE
        return tf.math.pow(1 - t, 4) * ((1 - sigma) * t0 + (t0 * sigma) * tf.math.cos(t * lamb * CYC))


class NoopAnnealingScheduler(AnnealingScheduler):
    def get_value(self, t):
        return 1.0


def create_random_inpaint_mask(batch, n_holes=4):
    """
    Applies random irregular masks to batch of character images.
    Returns masked batch and masks.

    :param batch: Tensor of shape (b, d, s, s, c)
    :param n_holes: Number of irregular holes to generate per mask
    :return: Tuple of (masked_batch, masks)
        - masked_batch: Tensor with holes applied to RGB channels
        - masks: Tensor of shape (b, s, s, 1) with holes
    """
    # Precompute grid once for the entire batch
    batch_size, number_of_domains, image_size = batch.shape[0], batch.shape[1], batch.shape[2]
    if n_holes == 0:
        # no holes, return original batch and empty masks
        # need to declare an empty grid tensor to match the expectation of the tf.function
        grid = tf.zeros([image_size, image_size, 2], dtype=tf.float32)
        return batch, tf.zeros([batch_size, image_size, image_size, 1])

    grid_i, grid_j = tf.meshgrid(tf.range(image_size, dtype=tf.float32),
                                 tf.range(image_size, dtype=tf.float32), indexing='ij')
    grid = tf.stack([grid_i, grid_j], axis=-1)  # Shape: [64, 64, 2]

    def generate_mask_for_character(example):
        """Generate single mask for all directions of one character"""
        # Combine alphas from all directions (union of character pixels)
        alphas = example[..., 3]  # Shape: [4, 64, 64]
        combined_alpha = tf.reduce_max(alphas, axis=0)  # Shape: [64, 64]

        # Find character pixels
        indices = tf.where(combined_alpha > 0.5)  # Shape: [n, 2]
        n_points = tf.shape(indices)[0]

        def create_mask():
            # Sample points from character pixels
            idx = tf.random.uniform(shape=[n_holes], minval=0, maxval=n_points, dtype=tf.int32)
            base_points = tf.gather(indices, idx)  # Shape: [n_holes, 2]
            base_points_float = tf.cast(base_points, tf.float32)

            # Add small random offsets
            offset = 0.  # tf.random.uniform(shape=[n_holes, 2], minval=-0.5, maxval=0.5, dtype=tf.float32)
            points = base_points_float + offset  # Shape: [n_holes, 2]

            # Generate random radii for holes
            radii = tf.random.uniform(shape=[n_holes], minval=2.0, maxval=9.0, dtype=tf.float32)
            squared_radii = tf.square(radii)  # Shape: [n_holes]

            # Compute distances from grid points to centers
            diff = grid[tf.newaxis, :, :, :] - points[:, tf.newaxis, tf.newaxis, :]  # Shape: [n_holes, 64, 64, 2]
            squared_distances = tf.reduce_sum(tf.square(diff), axis=-1)  # Shape: [n_holes, 64, 64]

            # Create mask by combining shapes
            mask_i = squared_distances < squared_radii[:, tf.newaxis, tf.newaxis]  # Shape: [n_holes, 64, 64]
            mask = tf.reduce_any(mask_i, axis=0)  # Shape: [64, 64]
            return tf.cast(mask, tf.float32)

        # Return zero mask if no character pixels
        return tf.cond(n_points > 0, create_mask, lambda: tf.zeros([image_size, image_size], dtype=tf.float32))

    masks_list = tf.map_fn(generate_mask_for_character, batch)

    # stack masks and expand dimensions
    masks = tf.stack(masks_list, axis=0)  # Shape: [batch_size, 64, 64]
    masks = tf.expand_dims(masks, axis=1)  # Add domain dimension: [batch_size, 1, 64, 64]
    masks = tf.expand_dims(masks, axis=-1)  # Add channel dimension: [batch_size, 1, 64, 64, 1]

    # replicate mask across all 4 directions
    masks = tf.tile(masks, [1, number_of_domains, 1, 1, 1])  # Shape: [batch_size, 4, 64, 64, 1]

    # replace the RGBA values of the batch where the mask is 1 with -1
    masked_batch = tf.where(masks == 1.0, -1.0, batch)  # Shape: [batch_size, 4, 64, 64, 4]

    return masked_batch, masks[:, 0, ...]  # return only one domain mask (e.g., first domain), as they are the same


class ConcatenateMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Concatenates an inpainting mask to the input images.
        :param inputs: Tuple of (images, masks), where:
            - images: Tensor of shape [b, d, s, s, c]
            - masks: Tensor of shape [b, s, s, 1]
        :return: Concatenated tensor of shape [b, d, s, s, c+1]
        """
        images, masks = inputs
        number_of_domains = images.shape[1]
        masks = tf.expand_dims(masks, axis=1)
        # masks (shape=[b, 1, s, s, 1])
        masks = tf.tile(masks, [1, number_of_domains, 1, 1, 1])
        # masks (shape=[b, d, s, s, 1])
        return tf.concat([images, masks], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     return input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4] + 1


def list_of_palettes_to_ragged_tensor(palettes):
    """
    Converts a list of palettes to a ragged tensor.
    :param palettes: List of palettes, where each palette is a 2D array of shape [k, c]
    :return: RaggedTensor of shape [b, (k), c]
    """
    # creates a flattened tensor of all colors
    values = tf.concat(palettes, axis=0)
    # creates a list of row lengths (number of colors) for each palette
    row_lengths = [tf.shape(palette)[0] for palette in palettes]

    return tf.RaggedTensor.from_row_lengths(values, row_lengths)


def scales_output_to_two_halves(scales_output):
    """
    Splits the output of the scales output of a generator into two halves along the batch dimension.
    :param scales_output: a list of tensors [scales] x shape=[b, d, h, w, c] or [b, d, h, w]
    :return: two lists (half_1, half_2), each having [scales] x shape=[b/2, d, h, w, c] or [b/2, d, h, w]
    """
    half_1 = []
    half_2 = []
    for tensor in scales_output:
        # split each tensor into 2 parts along batch dim
        split_tensors = tf.split(tensor, num_or_size_splits=2, axis=0)
        half_1.append(split_tensors[0])
        half_2.append(split_tensors[1])

    # convert lists to tensors
    return half_1, half_2


class InpaintMaskGenerator(ABC):
    """
    Abstract base class for inpainting modes.
    Each mode should implement the `apply` method.
    """
    @abstractmethod
    def apply(self, batch, t):
        """
        Applies the inpainting mode to the images with the given masks.
        :param batch: Tensor of shape [b, d, s, s, c]
        :param t: Current training progress (0.0 to 1.0)
        :return: Tuple of (masked_batch, masks)
            - masked_batch: Tensor with holes applied to RGB channels
            - masks: Tensor of shape [b, s, s, 1] with holes
        """
        pass


class NoopInpaintMaskGenerator(InpaintMaskGenerator):
    """
    No-op inpainting mode that returns the images unchanged.
    """
    def apply(self, batch, t):
        empty_mask = tf.zeros([batch.shape[0], batch.shape[2], batch.shape[3], 1], dtype=batch.dtype)
        return batch, empty_mask


class ConstantInpaintMaskGenerator(InpaintMaskGenerator):
    """
    Random inpainting mode that applies random irregular masks to the images.
    """
    def __init__(self, num_holes=4):
        self.num_holes = num_holes

    def apply(self, batch, t):
        """
        Applies random irregular masks to the batch of images.
        :param batch: Tensor of shape [b, d, s, s, c]
        :param t: Current training progress (0.0 to 1.0)
        :return: Tuple of (masked_batch, masks)
            - masked_batch: Tensor with holes applied to RGB channels
            - masks: Tensor of shape [b, s, s, 1] with holes
        """
        return create_random_inpaint_mask(batch, self.num_holes)


class RandomInpaintMaskGenerator(InpaintMaskGenerator):
    """
    Random inpainting mode that applies random irregular masks to the images.
    """
    def __init__(self, max_holes=4):
        self.max_holes = max_holes

    def apply(self, batch, t):
        """
        Applies random irregular masks to the batch of images.
        :param batch: Tensor of shape [b, d, s, s, c]
        :param t: Current training progress (0.0 to 1.0)
        :return: Tuple of (masked_batch, masks)
            - masked_batch: Tensor with holes applied to RGB channels
            - masks: Tensor of shape [b, s, s, 1] with holes
        """
        num_holes = tf.random.uniform([], minval=0, maxval=self.max_holes + 1, dtype=tf.int32)
        return create_random_inpaint_mask(batch, num_holes)


class CurriculumInpaintMaskGenerator(InpaintMaskGenerator):
    """
    Curriculum inpainting mode that applies masks to the images based on a curriculum.
    The curriculum is defined by the `curriculum` parameter, which is a list of tuples
    (n_holes, mask_probability).
    """
    def __init__(self, max_holes=4):
        self.max_holes = max_holes

    def apply(self, batch, t):
        """
        Applies curriculum-based inpainting to the batch of images.
        :param batch: Tensor of shape [b, d, s, s, c]
        :param t: Current training progress (0.0 to 1.0)
        :return: Tuple of (masked_batch, masks)
            - masked_batch: Tensor with holes applied to RGB channels
            - masks: Tensor of shape [b, s, s, 1] with holes
        """
        is_random = t >= 0.5
        num_holes = tf.random.uniform([], minval=0, maxval=self.max_holes + 1, dtype=tf.int32) \
            if is_random \
            else tf.cast(tf.math.floordiv(t, 0.5) * (self.max_holes + 1), tf.int32)
        return create_random_inpaint_mask(batch, num_holes)


# Custom weight initializer which is very similar to MSRInitializer (He/MSRA), but allows to set a custom gain which
# is useful for activation functions derived from ReLU, such as LeakyReLU This has been adapted from the R3GAN
# implementation: https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/R3GAN/Networks.py#L7
class MSRInitializer(keras.initializers.VarianceScaling):
    def __init__(self, activation_gain=1.0):
        super(MSRInitializer, self).__init__(
            scale=activation_gain, mode="fan_in", distribution="truncated_normal")


class AdversarialLoss(ABC):
    def __init__(self, number_of_domains, discriminator_scales, domain_specific_discriminators=True):
        self.number_of_domains = number_of_domains
        self.discriminator_scales = discriminator_scales
        self.domain_reductor = lambda x: tf.reduce_mean(x, axis=1) if not domain_specific_discriminators else x

    @abstractmethod
    def calculate_discriminator_loss(self, fake_logits, real_logits):
        """
        Calculates the adversarial loss between the real and fake images for the discriminator.
        The logits can have the following shape:
          - [d] x [ds] x [b, x, x, 1]
        The logits are reduced to [d, ds] => [d] if domain_specific_discriminators is True, else [ds] => []
        :param fake_logits: the output of the discriminator for the fake images
        :param real_logits: the output of the discriminator for the real images
        :return: the adversarial loss, on the discriminator perspective, and the real and fake terms
        """
        pass

    @abstractmethod
    def calculate_generator_loss(self, fake_logits, real_logits=None):
        """
        Calculates the adversarial loss between for the generator.
        The logits can have the following shapes:
          - [d] x [ds] x [b, x, x, 1] if both discriminator_scales > 1 and domain_specific_discriminators is True
          - [ds] x [b, x, x, 1] if discriminator_scales > 1
          - [b, x, x, 1] if discriminator_scales == 1
          - [ds] x [b, x, x, 1] if domain_specific_discriminators is True
        The logits are reduced to [d, ds] => [d] if domain_specific_discriminators is True, else [ds] => []
        :param fake_logits: the output of the discriminator for the fake images
        :param real_logits: the output of the discriminator for the real images, if necessary (otherwise None)
        :return: the adversarial loss on the generator perspective
        """
        pass


class LSGANLoss(AdversarialLoss):
    def __init__(self, number_of_domains, discriminator_scales, domain_specific_discriminators=True):
        super().__init__(number_of_domains, discriminator_scales, domain_specific_discriminators)
        self.loss = tf.keras.losses.MeanSquaredError()

    def calculate_generator_loss(self, fake_logits, real_logits=None):
        new_fake_logits = [[[] for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        for d in range(self.number_of_domains):
            for ds in range(self.discriminator_scales):
                new_fake_logits[d][ds] = self.loss(tf.ones_like(fake_logits[d][ds]), fake_logits[d][ds])
                # fake_logits (shape=[d] x [ds] x [b, x, x, c])
                new_fake_logits[d][ds] = tf.reduce_mean(new_fake_logits[d][ds])
                # fake_logits (shape=[d] x [ds])
        new_fake_logits = tf.reduce_mean(new_fake_logits, axis=1)
        # fake_logits (shape=[d])
        adversarial_loss = self.domain_reductor(new_fake_logits)
        # adversarial_loss (shape=[d] if domain_specific_discriminators else [])

        return adversarial_loss

    def calculate_discriminator_loss(self, fake_logits, real_logits):
        # shape=[d, ds] x [b, x, x, 1] => [d, ds] => [d]
        new_real_logits = [[[] for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        new_fake_logits = [[[] for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        for d in range(self.number_of_domains):
            for ds in range(self.discriminator_scales):
                new_real_logits[d][ds] = self.loss(tf.ones_like(real_logits[d][ds]), real_logits[d][ds])
                new_fake_logits[d][ds] = self.loss(tf.zeros_like(fake_logits[d][ds]), fake_logits[d][ds])
                # xxxx_logits (shape=[d] x [ds] x [b, x, x, c])
                new_real_logits[d][ds] = tf.reduce_mean(new_real_logits[d][ds])
                new_fake_logits[d][ds] = tf.reduce_mean(new_fake_logits[d][ds])
                # xxxx_logits (shape=[d] x [ds])

        new_real_logits = tf.reduce_mean(new_real_logits, axis=1)
        new_fake_logits = tf.reduce_mean(new_fake_logits, axis=1)
        adversarial_loss = [new_real_logits[d] + new_fake_logits[d] for d in range(self.number_of_domains)]
        # xxxx_logits (shape=[d])

        real_loss = self.domain_reductor(new_real_logits)
        fake_loss = self.domain_reductor(new_fake_logits)
        adversarial_loss = self.domain_reductor(adversarial_loss)
        # xxxx_logits (shape=[d] if domain_specific_discriminators else [])

        return adversarial_loss, real_loss, fake_loss


class RelativisticLoss(AdversarialLoss):
    def calculate_generator_loss(self, fake_logits, real_logits=None):
        role_inverted_discriminator_loss, _, _ = self.calculate_discriminator_loss(real_logits, fake_logits)
        return role_inverted_discriminator_loss

    def calculate_discriminator_loss(self, fake_logits, real_logits):
        relativistic_logits = [[0. for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        new_real_logits = [[[] for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        new_fake_logits = [[[] for _ in range(self.discriminator_scales)] for _ in range(self.number_of_domains)]
        for d in range(self.number_of_domains):
            for ds in range(self.discriminator_scales):
                relativistic_logits[d][ds] = real_logits[d][ds] - fake_logits[d][ds]
                # relativistic_logits ([d] x [ds] x shape=[b, x, x, 1])
                relativistic_logits[d][ds] = tf.reduce_mean(relativistic_logits[d][ds])
                # relativistic_logits ([d] x [ds])

                new_real_logits[d][ds] = tf.reduce_mean(real_logits[d][ds])
                new_fake_logits[d][ds] = tf.reduce_mean(fake_logits[d][ds])
                # xxxx_logits ([d] x [ds])

        relativistic_logits = tf.reduce_mean(relativistic_logits, axis=1)
        # relativistic_logits ([d])
        new_real_logits = tf.reduce_mean(new_real_logits, axis=1)
        new_fake_logits = tf.reduce_mean(new_fake_logits, axis=1)
        # xxxx_logits ([d])

        relativistic_logits = self.domain_reductor(relativistic_logits)
        # relativistic_logits (shape=[d] if domain_specific_discriminators else [])
        new_real_logits = self.domain_reductor(new_real_logits)
        new_fake_logits = self.domain_reductor(new_fake_logits)
        # xxxx_logits (shape=[d] if domain_specific_discriminators else [])
        adversarial_loss = tf.math.softplus(-relativistic_logits)
        return adversarial_loss, new_real_logits, new_fake_logits


class GradientPenalty(ABC):
    @abstractmethod
    def __call__(self, discriminators, discriminators_input):
        pass


class NoopGradientPenalty(GradientPenalty):
    def __call__(self, discriminators, discriminators_input):
        """
        No-op gradient penalty that returns zero for each discriminator.
        :param discriminators: A list of 1 or more discriminators
        :param discriminators_input: Input images to the discriminator with shape [d] x [b, s, s, c]
        :return: List of zeros for each discriminator
        """
        return tf.constant(0.0)


class ZeroCenteredGradientPenalty(GradientPenalty):
    def __call__(self, discriminator, discriminator_input):
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(discriminator_input)
            output = listify(discriminator(discriminator_input, training=True))
            output = tf.reduce_mean([tf.reduce_sum(output) for output in output])

        gp_grads = gp_tape.gradient(output, discriminator_input)
        r_penalty = tf.reduce_sum(tf.square(gp_grads))

        del gp_tape

        return r_penalty


class FiLMLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        self.gamma = layers.Dense(input_shape[0][-1], use_bias=False)
        self.beta = layers.Dense(input_shape[0][-1], use_bias=False)

    def call(self, inputs):
        x, condition = inputs
        gamma = self.gamma(condition)[:, None, None, :]
        beta = self.beta(condition)[:, None, None, :]
        return gamma * x + beta


class AnyNonZeroPixelDetector(layers.Layer):
    """
    A layer that checks whether each image in the input batch is composed of all zeros. In
    case it is, it returns a zero scalar; otherwise, it returns a one scalar.
    The output has the same rank as the input, with shape (input_shape[0], 1, 1...)
    """
    def call(self, inputs):
        mask = tf.reduce_all(tf.equal(inputs, 0.0), axis=[1, 2, 3], keepdims=True)
        return 1. - tf.cast(mask, tf.float32)

class AdaIN(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_shape, condition_shape = input_shape
        num_channels = feature_shape[-1]
        
        # Initialize dense layers for gamma and beta with zeros (for kernel) and specific biases
        self.gamma_dense = layers.Dense(
            num_channels, 
            kernel_initializer='zeros',
            bias_initializer='ones'
        )
        self.beta_dense = layers.Dense(
            num_channels,
            kernel_initializer='zeros',
            bias_initializer='zeros'
        )

    def call(self, inputs):
        x, condition = inputs
        
        # Calculate mean and variance along spatial dimensions (H, W)
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std = tf.sqrt(variance + 1e-5)
        
        # Normalize the input
        normalized = (x - mean) / std
        
        # Project the condition to get gamma and beta
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)
        
        # Reshape gamma and beta to [batch_size, 1, 1, num_channels]
        gamma = gamma[:, tf.newaxis, tf.newaxis, :]
        beta = beta[:, tf.newaxis, tf.newaxis, :]
        
        return gamma * normalized + beta

# Lots of layers and code related to R3GAN, used by Sprite and ReMIC
# BiasActivation is a way to fuse the bias addition with the LeakyReLU activation, which improves performance
# especially in pytorch, which might have more trouble fusing it as an optimization compared to TensorFlow (as TF has
# graph mode and more optimizations in place).
# Hence, we rely on keeping the bias addition in the Linear and Conv2D layers.
class R3GANResidualBlock(layers.Layer):
    LeakyReLUGain = tf.math.sqrt(2. / (1. + 0.2 ** 2.))

    def __init__(self, variance_scaling_parameter, **kwargs):
        super().__init__(**kwargs)
        num_linear_layers = 3
        activation_gain = self.LeakyReLUGain * variance_scaling_parameter ** (-1 / (2 * num_linear_layers - 2))
        self.kernel_initializer = MSRInitializer(activation_gain)
        self.linear1 = None
        self.linear2 = None
        self.linear3 = None
        self.activation = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        cardinality = input_channels // 8
        expanded_channels = input_channels * 2

        # 1x1 convolution from input_channels to expanded_channels
        self.linear1 = self.create_convolution(expanded_channels, 1)
        # 3x3 convolution from expanded_channels to expanded_channels (reverse bottleneck),
        # with groups equal to cardinality
        self.linear2 = self.create_convolution(expanded_channels, 3, groups=cardinality)
        # self.linear2 = self.create_convolution(expanded_channels, 3)
        # self.linear2 = self.create_convolution(expanded_channels, 3, groups=expanded_channels)
        # 1x1 convolution from expanded_channels back to input_channels
        self.linear3 = self.create_convolution(input_channels, 1, use_bias=False)

        # plain 0.2 lre
        self.activation = tf.keras.layers.LeakyReLU(negative_slope=0.2)

    def call(self, x):
        y = self.activation(self.linear1(x))
        y = self.activation(self.linear2(y))
        y = self.linear3(y)
        return x + y

    def compute_output_shape(self, input_shape):
        return input_shape

    def create_convolution(self, filters, kernel_size, groups=1, use_bias=True):
        return layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same", use_bias=use_bias,
                             kernel_initializer=self.kernel_initializer, bias_initializer="zeros", groups=groups)

class R3GANResidualBlockConditional(layers.Layer):
    LeakyReLUGain = tf.math.sqrt(2. / (1. + 0.2 ** 2.))

    def __init__(self, variance_scaling_parameter, condition_layer="adain", **kwargs):
        super().__init__(**kwargs)
        num_linear_layers = 3
        activation_gain = self.LeakyReLUGain * variance_scaling_parameter ** (-1 / (2 * num_linear_layers - 2))
        self.kernel_initializer = MSRInitializer(activation_gain)
        self.linear1 = None
        self.linear2 = None
        self.linear3 = None
        self.activation = None
        self.conditioner = None
        self.condition_layer = condition_layer

    def build(self, input_shape):
        input_channels = input_shape[0][-1]
        cardinality = input_channels // 8
        expanded_channels = input_channels * 2

        # 1x1 convolution from input_channels to expanded_channels
        self.linear1 = self.create_convolution(expanded_channels, 1)
        # 3x3 convolution from expanded_channels to expanded_channels (reverse bottleneck),
        # with groups equal to cardinality
        self.linear2 = self.create_convolution(expanded_channels, 3, groups=cardinality)
        # self.linear2 = self.create_convolution(expanded_channels, 3)
        # self.linear2 = self.create_convolution(expanded_channels, 3, groups=expanded_channels)
        # 1x1 convolution from expanded_channels back to input_channels
        self.linear3 = self.create_convolution(input_channels, 1, use_bias=False)

        if self.condition_layer not in ["adain", "film"]:
            raise ValueError(f"Unknown condition_layer: {self.condition_layer}. Supported: 'adain', 'film'")
        elif self.condition_layer == "adain":
            self.conditioner = AdaIN()
        elif self.condition_layer == "film":
            self.conditioner = FiLMLayer()

        # plain 0.2 lre
        self.activation = layers.LeakyReLU(negative_slope=0.2)

    def call(self, inputs):
        x, condition = inputs
        y = self.linear1(x)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.conditioner([y, condition])
        y = self.activation(y)
        y = self.linear3(y)
        return x + y

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def create_convolution(self, filters, kernel_size, groups=1, use_bias=True):
        return layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same", use_bias=use_bias,
                             kernel_initializer=self.kernel_initializer, bias_initializer="zeros", groups=groups)


def R3GANResidualBlockBuilder(is_conditional, condition_layer="adain"):
    def builder(variance_scaling_parameter, **kwargs):
        if is_conditional:
            return R3GANResidualBlockConditional(variance_scaling_parameter, **kwargs)
        else:
            return R3GANResidualBlock(variance_scaling_parameter, **kwargs)

    return builder

def create_lowpass_kernel():
    w = [1., 2., 1.]
    w_padded = tf.pad(w, [[1, 1]], constant_values=0)
    kernel_1d = w_padded[:-1] + w_padded[1:]
    kernel_1d = tf.reshape(kernel_1d, [1, -1])
    kernel_2d = tf.matmul(tf.transpose(kernel_1d), kernel_1d)
    kernel_2d = kernel_2d / tf.reduce_sum(kernel_2d)
    return kernel_2d

class InterpolativeUpsampler(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lowpass_kernel = create_lowpass_kernel()
        self.built_kernel = None

    def build(self, input_shape):
        kernel = self.lowpass_kernel[..., tf.newaxis, tf.newaxis]
        self.built_kernel = self.add_weight(
            name="lowpass_filter",
            shape=kernel.shape,
            initializer=tf.constant_initializer(kernel.numpy()),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        x = inputs

        input_shape = tf.shape(x)
        batch_size, image_size, channels = input_shape[0], input_shape[-2], input_shape[-1]
        new_size = image_size * 2
        y = tf.nn.conv2d_transpose(
            x,
            filters=tf.tile(self.built_kernel, [1, 1, channels, channels]),
            output_shape=[batch_size, new_size, new_size, channels],
            strides=[1, 2, 2, 1],
        )
        y = y * 4.0
        return y

class InterpolativeDownsampler(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lowpass_kernel = create_lowpass_kernel()
        self.built_kernel = None

    def build(self, input_shape):
        kernel = self.lowpass_kernel[..., tf.newaxis, tf.newaxis]
        self.built_kernel = self.add_weight(
            name="lowpass_filter",
            shape=kernel.shape,
            initializer=tf.constant_initializer(kernel.numpy()),
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        channels = tf.shape(x)[-1]
        y = tf.nn.depthwise_conv2d(
            x,
            filter=tf.tile(self.built_kernel, [1, 1, channels, 1]),
            strides=[1, 2, 2, 1],
            padding="SAME",
            data_format="NHWC"
        )
        return y


class UpsampleLayer(layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.resampler = None
        self.linear = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        current_size = input_shape[-2]
        new_size = current_size * 2
        # self.resampler = layers.Resizing(new_size, new_size, interpolation="bilinear")
        # self.resampler = InterpolativeUpsampler()
        # self.resampler = layers.Lambda(lambda x: tf.image.resize(x, (new_size, new_size), method="bilinear"), output_shape=(new_size, new_size, input_channels))
        self.resampler = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        if input_channels != self.output_channels:
            self.linear = layers.Conv2D(self.output_channels, 1, strides=1, padding="same",
                                        use_bias=False, kernel_initializer=MSRInitializer(1.0))

    def call(self, x):
        if self.linear is not None:
            x = self.linear(x)
        x = self.resampler(x)
        return x


class DownsampleLayer(layers.Layer):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.resampler = None
        self.linear = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        current_size = input_shape[-2]
        new_size = current_size // 2
        # self.resampler = layers.Resizing(new_size, new_size, interpolation="bilinear")
        self.resampler = InterpolativeDownsampler()
        # self.resampler = layers.Lambda(lambda x: tf.image.resize(x, (new_size, new_size), method="bilinear"), output_shape=(new_size, new_size, input_channels))
        # self.resampler = layers.AveragePooling2D(2)
        if input_channels != self.output_channels:
            self.linear = layers.Conv2D(self.output_channels, 1, strides=1, padding="same", use_bias=False,
                                        kernel_initializer=MSRInitializer(1.0))

    def call(self, x):
        x = self.resampler(x)
        if self.linear is not None:
            x = self.linear(x)
        return x

    def compute_output_shape(self, input_shape):
        current_size = input_shape[-2]
        new_size = current_size // 2
        return input_shape[0], new_size, new_size, self.output_channels


class GenerativeBasis(layers.Layer):
    def __init__(self, output_channels, use_1x1=False):
        super().__init__()
        self.output_channels = output_channels
        if use_1x1:
            self.linear = layers.Conv2D(output_channels, 1, strides=1, padding="same",
                                        kernel_initializer=MSRInitializer(1.0))
        else:
            self.linear = layers.Dense(output_channels, use_bias=False, kernel_initializer=MSRInitializer(1.0))
        self.basis = None

    def build(self, input_shape):
        spatial_shape = input_shape[-2]
        self.basis = self.add_weight(shape=(spatial_shape, spatial_shape, self.output_channels), initializer="random_normal",
                                     trainable=True)

    def call(self, x):
        linear_output = self.linear(x)
        return self.basis[tf.newaxis, ...] * linear_output


class DiscriminativeBasis(layers.Layer):
    def __init__(self, input_channels, output_dimension):
        super().__init__()
        self.basis = layers.Conv2D(
            input_channels, kernel_size=4, strides=1, padding="valid", groups=input_channels, use_bias=False,
            kernel_initializer=MSRInitializer(1.0)
        )
        self.linear = layers.Dense(output_dimension, use_bias=False, kernel_initializer=MSRInitializer(1.0))
        self.flatten = layers.Flatten()

    def call(self, x):
        x = self.basis(x)
        x = self.flatten(x)
        return self.linear(x)


class UpsampleStage(layers.Layer):
    def __init__(self, output_channels, variance_scaling_parameter,
                 is_initial_stage=False, is_conditional=False, condition_layer="adain"):
        super().__init__()
        self.output_channels = output_channels
        self.variance_scaling_parameter = variance_scaling_parameter
        self.is_initial_stage = is_initial_stage
        self.is_conditional = is_conditional
        self.transition_layer = None
        self.resblock_1 = None
        self.resblock_2 = None
        self.ResidualBlockConstructor = R3GANResidualBlockBuilder(is_conditional, condition_layer)
        self.resblock_supplier = NParamsSupplier(2 if is_conditional else 1, delistify=True)
        self.inputs_padder = lambda inputs: inputs if is_conditional else (inputs, None)

    def build(self, input_shape):
        if self.is_initial_stage:
            self.transition_layer = GenerativeBasis(self.output_channels)
        else:
            self.transition_layer = UpsampleLayer(self.output_channels)

        self.resblock_1 = self.ResidualBlockConstructor(self.variance_scaling_parameter)
        self.resblock_2 = self.ResidualBlockConstructor(self.variance_scaling_parameter)

    def call(self, x):
        x, condition = self.inputs_padder(x)
        x = self.transition_layer(x)
        x = self.resblock_1(self.resblock_supplier(x, condition))
        x = self.resblock_2(self.resblock_supplier(x, condition))
        return x

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0] if self.is_conditional else input_shape
        current_size = input_shape[-2]
        new_size = current_size * 2
        if self.is_initial_stage:
            return input_shape[0], current_size, current_size, self.output_channels
        else:
            return input_shape[0], new_size, new_size, self.output_channels


class DownsampleStage(layers.Layer):
    def __init__(self, output_channels, variance_scaling_parameter,
                 is_last_stage=False, is_conditional=False, condition_layer="adain"):
        super().__init__()
        self.output_channels = output_channels
        self.variance_scaling_parameter = variance_scaling_parameter
        self.is_last_stage = is_last_stage
        self.is_conditional = is_conditional
        self.condition_layer = condition_layer
        self.transition_layer = None
        self.resblock_1 = None
        self.resblock_2 = None
        self.ResidualBlockConstructor = R3GANResidualBlockBuilder(is_conditional, condition_layer)
        self.resblock_supplier = NParamsSupplier(2 if is_conditional else 1, delistify=True)
        self.inputs_padder = lambda inputs: inputs if is_conditional else (inputs, None)

    def build(self, input_shape):
        input_channels = input_shape[0][-1] if self.is_conditional else input_shape[-1]
        if self.is_last_stage:
            self.transition_layer = DiscriminativeBasis(input_channels, self.output_channels)
        else:
            self.transition_layer = DownsampleLayer(self.output_channels)

        self.resblock_1 = self.ResidualBlockConstructor(self.variance_scaling_parameter)
        self.resblock_2 = self.ResidualBlockConstructor(self.variance_scaling_parameter)

    def call(self, x):
        x, condition = self.inputs_padder(x)
        x = self.resblock_1(self.resblock_supplier(x, condition))
        x = self.resblock_2(self.resblock_supplier(x, condition))
        x = self.transition_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0] if self.is_conditional else input_shape
        current_size = input_shape[-2]
        new_size = current_size // 2
        if self.is_last_stage:
            return input_shape[0], current_size, current_size, self.output_channels
        else:
            return input_shape[0], new_size, new_size, self.output_channels


def check_nans(tensor, name):
    # return tensor
    return tf.debugging.check_numerics(tensor, f"NaN in {name}")


class DetectPresentImages(layers.Layer):
    def __init__(self, **kwargs):
        super(DetectPresentImages, self).__init__(**kwargs)

    def call(self, x):
        # x (shape=[b, d, s, s, c])
        non_zero = tf.math.not_equal(x, 0.0)
        has_non_zero = tf.reduce_any(non_zero, axis=[2, 3, 4])
        return tf.cast(has_non_zero, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]  # [b, d]


class ShapeLogger(tf.keras.layers.Layer):
    def __init__(self, name="shape_logger", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        # tf.shape works in graph mode and eager mode
        tf.print(f"[{self.name}] shape:", tf.shape(inputs))
        return inputs
    
