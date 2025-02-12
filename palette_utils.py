import tensorflow as tf
from tensorflow import RaggedTensorSpec

import dataset_utils


def get_invalid_color(channels):
    return tf.constant([32768] * channels)


def extract_palette(image):
    """
    Extracts the unique colors from an image (3D tensor)
    Parameters
    ----------
    image a 3D tensor with shape (height, width, channels). Values should be int32 inside [0, 255]. Can also be a
    batched 4D tensor, in which case it looks for the combined (single) palette for all such images (this is useful
    if we have front/right/back/left of a sprite and want a combined palette).

    Returns a tensor of colors (RGB) representing an image's palette
    -------
    """
    channels = tf.shape(image)[-1]

    # incoming image shape: (s, s, channels)
    # reshaping to: (s*s, channels)
    image = tf.cast(image, tf.int8)
    image = tf.reshape(image, [-1, channels])

    # colors are sorted as they appear in the image sweeping from top-left to bottom-right
    colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])
    return colors


def batch_extract_palette(images):
    """
    Extracts the palette of each image in the batch, returning a ragged tensor of shape [b, (colors), c]
    :param images:
    :return:
    """
    images = dataset_utils.denormalize(images)
    images = tf.cast(images, tf.int32)

    palettes_ragged = tf.map_fn(fn=extract_palette, elems=images,
                                fn_output_signature=RaggedTensorSpec(
                                    ragged_rank=0,
                                    dtype=tf.int32))
    palettes = tf.RaggedTensor.to_tensor(palettes_ragged, default_value=get_invalid_color(channels))
    palettes = tf.cast(palettes, tf.float32)
    palettes = dataset_utils.normalize(palettes)

    return palettes


def extract_palette_ragged(image):
    """
    Extracts the unique colors from an image (3D tensor) -- returns a ragged tensor with shape [1, (colors), c]
    :params: image: a 3D tensor with shape (height, width, channels). Values should be float32 inside [-1, 1].
    """
    channels = tf.shape(image)[-1]

    # incoming image shape: (s, s, channels)
    # reshaping to: (s*s, channels)
    image = tf.cast(image, tf.int8)
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
    :param images: batch of images: [b, s, s, c]
    :return:
    """
    images = dataset_utils.denormalize(images)
    images = tf.cast(images, tf.int8)
    channels = images.shape[-1]

    palettes_ragged = tf.map_fn(fn=extract_palette_ragged, elems=images,
                                fn_output_signature=RaggedTensorSpec(
                                    shape=(1, None, channels),
                                    ragged_rank=1,
                                    dtype=tf.int8))

    palettes_ragged = palettes_ragged.merge_dims(1, 2)
    palettes_ragged = tf.cast(palettes_ragged, tf.float32)
    palettes_ragged = dataset_utils.normalize(palettes_ragged)

    return palettes_ragged


# Based off: https://stackoverflow.com/a/43839605/1783793
# Idea: D = A²x1nb + B²x1na - 2xABt
@tf.function
def l2_between_points(a, b):
    # A.shape == (b, n_a, 4)
    # B.shape == (b, n_b, 4)
    n_a = tf.shape(a)[1]
    n_b = tf.shape(b)[1]

    # A² with each channel squared then summed: [b, n_a, 1]
    a2 = tf.reduce_sum(a ** 2, -1, keepdims=True)
    # B² with each channel squared then summed: [b, n_b, 1]
    b2 = tf.reduce_sum(b ** 2, -1, keepdims=True)
    # AxBt, i.e., (n_a, 4) x (4, n_b):          [b, n_a, n_b]
    a_times_b = tf.matmul(a, b, transpose_b=True)

    # A² but repeated to be of shape [b, n_a, n_b]
    a2 = tf.matmul(a2, tf.ones([1, n_b]))
    # B² transposed and in the shape [b, n_a, n_b]
    b2 = tf.matmul(b2, tf.ones([1, n_a]))
    b2 = tf.transpose(b2, perm=[0, 2, 1])

    return tf.sqrt(tf.maximum(0., a2 + b2 - 2. * a_times_b))


def calculate_palette_loss(palettes, images):
    """
    Calculates the palette loss for a batch of images. It is calculated for each image as the mean of the distance
    of each pixel to the closest color in its intended palette. The loss value for each image is then averaged for
    all images in the batch.
    :param palettes: batch of palettes (ragged tensor): [b, (c), 4]
    :param images: batch of images: [b, s, s, 4]
    :return: a scalar with the mean (over batch) of the palette losses.
    """
    images_shape = tf.shape(images)
    batch_size, image_size, channels = images_shape[0], images_shape[1], images_shape[3]

    # flattens the images in the batch: [b, s*s, 4]
    batch_pixels = tf.reshape(images, [batch_size, image_size * image_size, channels])
    # finds the distance of each pixel to each color in the palette: [b, (c), s*s]
    distances_to_palette = l2_between_points(palettes, batch_pixels)
    # finds the closest color from the palette for each pixel: [b, s*s]
    closest_color_indices = tf.argmin(distances_to_palette, axis=1)
    # finds the distance each pixel is from its closest color in the palette: [b, s*s, 1]
    distances_to_closest = batch_pixels - tf.gather(palettes, closest_color_indices, batch_dims=1)
    # calculates the loss
    # palette_loss = tf.reduce_mean(tf.reduce_sum(tf.square(distances_to_closest), axis=[1, 2]))
    palette_loss = tf.reduce_mean(tf.square(distances_to_closest))

    return palette_loss


def main():
    # image.shape == (4, 4, 1)
    image1 = tf.constant([
        [[255], [120], [127], [0]],
        [[127], [100], [255], [0]],
        [[100], [127], [0], [127]],
        [[100], [255], [100], [0]],
    ])
    image2 = tf.constant([
        [[200], [0], [10], [40]],
        [[200], [0], [20], [30]],
        [[50], [35], [30], [20]],
        [[50], [35], [40], [10]],
    ])
    image1 = tf.tile(image1, [1, 1, 4])
    image2 = tf.tile(image2, [1, 1, 4])
    # palette1 = extract_palette(image1)
    # palette2 = extract_palette(image2)
    # palette1 = tf.cast(palette1, "float32")
    # palette1 = palette1 / 255.
    # palette2 = tf.cast(palette2, "float32")
    # palette2 = palette2 / 255.
    image = tf.stack([image1, image2])
    palette_ragged = batch_extract_palette(image)
    palette = tf.RaggedTensor.to_tensor(palette_ragged, default_value=tf.constant([32768, 32768, 32768, 32768]))
    palette = tf.cast(palette, "float32")
    palette /= 255.
    image = tf.cast(image, "float32")
    image /= 255.
    # image1 = tf.cast(image1, "float32")
    # image1 = image1 / 255.
    # image2 = tf.cast(image2, "float32")
    # image2 = image2 / 255.

    jitter = tf.random.uniform(tf.shape(image1), maxval=0.8, seed=42.)
    # image = tf.stack([image1, image2])

    # palette = tf.map_fn(extract_palette, image)
    # palette = tf.ragged.stack([palette1, palette2])
    # palette = tf.RaggedTensor.from_row_lengths(tf.concat([palette1, palette2], axis=0), [tf.shape(palette1)[0], tf.shape(palette2)[0]])
    # palette = batch_extract_palette(image)

    # palette = tf.reshape(palette, [2, -1, tf.shape(palette1)[-1]])
    print("palette (ragged)", palette)
    print("tf.shape(palette)", tf.shape(palette))

    # with tf.GradientTape(persistent=True) as tape:
    #     generated_image = image + jitter
    #     tape.watch(generated_image)
    #
    #     # flattens the image
    #     pixels = tf.reshape(generated_image, [2, -1, 1])
    #     # finds the distance of each pixel to each color in the palette
    #     distances_to_palette = l2_between_points(palette, pixels)
    #     # finds the closest color from the palette for each pixel
    #     closest_color_indices = tf.argmin(distances_to_palette, axis=0)
    #     # finds the distance each pixel is from its closest color in the palette
    #     distances_to_closest = pixels - tf.gather(palette, closest_color_indices)
    #     # calculates the loss
    #     palette_loss = tf.reduce_mean(tf.square(distances_to_closest))
    #
    # print("palette", palette)
    # print("watched_variables", [var.name for var in tape.watched_variables()])
    # grads = tape.gradient(palette_loss, generated_image)
    # print("grads", grads)
    #
    # print("image", image)
    # print("generated_image", generated_image)
    # print("pixels", pixels)
    # print("distances_to_palette", distances_to_palette)
    # print("closest_color_indices", closest_color_indices)
    # print("distances_to_closest", distances_to_closest)
    # print("palette_loss", palette_loss)

    loss = calculate_palette_loss(palette, image + jitter)
    print("loss: image + jitter", loss)
    loss = calculate_palette_loss(palette, image)
    print("loss: image", loss)


if __name__ == "__main__":
    main()
