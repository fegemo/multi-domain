import tensorflow as tf
from tensorflow import RaggedTensorSpec

import dataset_utils


def get_invalid_color(channels):
    return tf.constant([32768] * channels)


def extract_palette(image):
    """
    Extracts the unique colors from an image (3D tensor)
    :params: image: a 3D tensor with shape (height, width, channels). Values should be inside [0, 255].
    :returns: a 2D tensor with shape (num_colors, channels) with the palette colors as uint8 inside [0, 255]
    """
    channels = tf.shape(image)[-1]

    # incoming image shape: (s, s, channels)
    # reshaping to: (s*s, channels)
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, [-1, channels])

    # colors are sorted as they appear in the image sweeping from top-left to bottom-right
    colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])
    return colors


def batch_extract_palette(images):
    """
    Extracts the palette of each image in the batch, returning a regular tensor of shape [b, max_colors, c]. The
    palettes are padded with the invalid color [32768, 32768, 32768, 32768] to have the same number of colors.
    :param images: batch of images: [b, s, s, c] with values inside [-1, 1]
    :return: a tensor with shape [b, max_colors, c] with the palette colors as float32 inside [-1, 1]
    """
    channels = tf.shape(images)[-1]
    images = dataset_utils.denormalize(images)
    images = tf.cast(images, tf.uint8)

    palettes_ragged = tf.map_fn(fn=extract_palette, elems=images,
                                fn_output_signature=RaggedTensorSpec(
                                ragged_rank=0,
                                dtype=tf.uint8))
    palettes = tf.RaggedTensor.to_tensor(palettes_ragged, default_value=get_invalid_color(channels))
    palettes = tf.cast(palettes, tf.float32)
    palettes = dataset_utils.normalize(palettes)

    return palettes


def extract_palette_ragged(image):
    """
    Extracts the unique colors from an image (3D tensor) -- returns a ragged tensor with shape [1, (colors), c]
    :params: image: a 3D tensor with shape (height, width, channels). Values should be inside [0, 255].
    :returns: a ragged tensor with shape [1, (num_colors), channels] with the palette colors as uint8 inside [0, 255]
    """
    channels = tf.shape(image)[-1]

    # incoming image shape: (s, s, channels)
    # reshaping to: (s*s, channels)
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, [-1, channels])

    # colors are sorted as they appear in the image sweeping from top-left to bottom-right
    colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])
    colors = tf.reshape(colors, [-1, channels])
    number_of_colors = tf.shape(colors)[0]
    # turns colors (a regular [num_colors, channels] tensor into a ragged tensor [1, (num_colors), channels])
    # this is necessary for an outer map_fn to call this function and have a ragged tensor as output
    colors = tf.RaggedTensor.from_row_lengths(values=colors, row_lengths=[number_of_colors])
    return colors

@tf.function
def batch_extract_palette_ragged(images):
    """
    Extracts the palette of each image in the batch, returning a ragged tensor of shape [b, (colors), c]
    :param images: batch of images: [b, s, s, c] with values inside [-1, 1]
    :return: a ragged tensor with shape [b, (colors), c] with the palette colors as float32 inside [-1, 1]
    """
    images = dataset_utils.denormalize(images)
    images = tf.cast(images, tf.uint8)
    channels = images.shape[-1]

    palettes_ragged = tf.map_fn(fn=extract_palette_ragged, elems=images,
                                fn_output_signature=RaggedTensorSpec(
                                    shape=(1, None, channels),
                                    ragged_rank=1,
                                    dtype=tf.uint8))

    palettes_ragged = palettes_ragged.merge_dims(1, 2)
    palettes_ragged = tf.cast(palettes_ragged, tf.float32)
    palettes_ragged = dataset_utils.normalize(palettes_ragged)

    return palettes_ragged


# Can receive a ragged palettes tensor (i.e., not filled with invalid colors)
def calculate_palette_loss(images, palettes):
    """
    Computes a loss term penalizing images for colors distant from the intended palette.

    :param images: A Tensor of shape [batch, size, size, channels] representing generated images.
    :param palettes: A Tensor of shape [batch, num_colors, channels] containing color palettes.
    :returns: A scalar Tensor representing the computed loss.
    """
    # Reshape images to [batch, num_pixels, channels]
    batch_size = tf.shape(images)[0]
    size = tf.shape(images)[1]
    images_reshaped = tf.reshape(images, [batch_size, size * size, -1])

    # Expand dimensions for broadcasting with the palette
    images_expanded = images_reshaped[:, :, tf.newaxis, :]  # [batch, num_pixels, 1, channels]
    palettes_expanded = palettes[:, tf.newaxis, :, :]  # [batch, 1, num_colors, channels] (ragged)

    # Compute squared differences between each pixel and palette color
    squared_diffs = tf.square(images_expanded - palettes_expanded)
    squared_distances = tf.reduce_sum(squared_diffs, axis=-1)  # [batch, num_pixels, num_colors] (ragged)

    # Find the minimum distance for each pixel
    min_distances = tf.reduce_min(squared_distances, axis=-1)  # [batch, num_pixels]

    # Average all minimum distances to get the final loss
    loss = tf.reduce_mean(min_distances)

    return loss

