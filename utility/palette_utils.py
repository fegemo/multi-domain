import tensorflow as tf
from tensorflow import RaggedTensorSpec

from . import dataset_utils


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


@tf.function
def calculate_palette_coverage_loss(images, palettes, temperature=0.1):
    """
    Calculates the palette coverage loss for a batch of images and their corresponding palettes.
    Colors that are in the palette but that have not been used in the image are penalized.

    Args:
        images (tf.Tensor): A batch of images with shape [batch_size, height, width, channels] in the [-1, 1] domain.
        palettes (tf.RaggedTensor): A batch of color palettes with shape [batch_size, (num_colors), channels] in the
            [-1, 1] domain.
        temperature (float): Controls the sharpness of the soft assignment. Lower values make the assignment sharper.

    Returns:
        tf.Tensor: A scalar loss value indicating how well the images cover their respective palettes averaged 4
        over the batch.
    """
    # Convert palettes to a dense tensor for easier processing
    palettes_dense = palettes.to_tensor(default_value=-100.0)  # Use -100 as a placeholder for missing colors

    # Get the shape of the palettes tensor
    batch_size, max_num_colors, channels = palettes_dense.shape

    # Reshape images to [batch_size, height * width, 3] to compare each pixel with the palette
    images_flat = tf.reshape(images, [batch_size, -1, channels])

    # Compute the pairwise squared differences between each pixel and each color in the palette
    # Shape: [batch_size, height * width, max_num_colors]
    diff = tf.reduce_sum(tf.square(images_flat[:, :, tf.newaxis, :] - palettes_dense[:, tf.newaxis, :, :]), axis=-1)

    # Compute soft assignment probabilities using softmin
    # Shape: [batch_size, height * width, max_num_colors]
    soft_assign = tf.nn.softmax(-diff / (temperature + 1e-8), axis=-1)

    # Create a mask to ignore the placeholder values in the palettes
    valid_mask = tf.reduce_all(palettes_dense != -100.0, axis=-1)  # Shape: [batch_size, max_num_colors]

    # Compute the total probability that each palette color is assigned to any pixel
    # Shape: [batch_size, max_num_colors]
    total_prob = tf.reduce_sum(soft_assign, axis=1)

    # Compute the coverage loss: penalize palette colors with low total probability
    # Shape: [batch_size, max_num_colors]
    coverage_loss = tf.maximum(1.0 - total_prob, 0.0)  # Loss is 1 if total_prob is 0, and 0 if total_prob is 1

    # Mask out invalid palette colors
    coverage_loss = tf.where(valid_mask, coverage_loss, tf.zeros_like(coverage_loss))

    # Sum the coverage loss over all valid palette colors
    total_loss = tf.reduce_sum(coverage_loss, axis=-1)

    # Normalize the loss by the number of valid colors in each palette
    num_valid_colors = tf.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1)
    normalized_loss = total_loss / (num_valid_colors + 1e-8)  # Add epsilon to avoid division by zero

    # Return the mean loss over the batch
    return tf.reduce_mean(normalized_loss)


def count_unused_palette_colors(images, palettes, threshold=1e-3):
    """
    Counts the number of colors in the palettes that are not used in the corresponding images.
    This is done as a debugging tool to check how many colors are being used. It is not necessarily differentiable.

    Args:
        images (tf.Tensor): A tensor of shape `[batch, size, size, channels]` representing a batch of images.
        palettes (tf.RaggedTensor): A ragged tensor of shape `[batch, (num_colors), channels]` representing a
            batch of color palettes.
        threshold (float): A small value to determine "closeness" to a color in the palette.

    Returns:
        tf.Tensor: An integer count of unused colors from the palettes for each image in the batch.
    """
    # Reshape images to [batch, num_pixels, channels]
    batch_size = tf.shape(images)[0]
    num_pixels = tf.shape(images)[1] * tf.shape(images)[2]
    images_reshaped = tf.reshape(images, [batch_size, num_pixels, tf.shape(images)[-1]])  # [batch, num_pixels, channels]

    # Convert the palettes RaggedTensor to a dense tensor
    palettes_dense = palettes.to_tensor(default_value=-100.0)  # [batch, max_num_colors, channels]

    # Create a mask to identify valid colors
    valid_mask = tf.reduce_any(palettes_dense != -100.0, axis=-1)  # [batch, max_num_colors]

    # Compute squared differences between each pixel and each palette color
    images_expanded = tf.expand_dims(images_reshaped, axis=2)  # [batch, num_pixels, 1, channels]
    palettes_expanded = tf.expand_dims(palettes_dense, axis=1)  # [batch, 1, max_num_colors, channels]
    squared_diffs = tf.square(images_expanded - palettes_expanded)  # [batch, num_pixels, max_num_colors, channels]
    squared_distances = tf.reduce_sum(squared_diffs, axis=-1)  # [batch, num_pixels, max_num_colors]

    # Check if any pixel is close enough to each palette color
    color_matches = tf.reduce_any(squared_distances <= threshold, axis=1)  # [batch, max_num_colors]

    # Count the number of unused colors for each image in the batch
    unused_colors = tf.logical_not(color_matches)  # Invert matches to find unused colors
    unused_colors = tf.logical_and(unused_colors, valid_mask)  # Mask out invalid colors
    unused_count = tf.reduce_sum(tf.cast(unused_colors, tf.int32), axis=-1)  # Count unused colors for each image

    return unused_count
