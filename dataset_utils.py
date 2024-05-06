import os

import tensorflow as tf


# Some images have transparent pixels with colors other than black
# This function turns all transparent pixels to black
# TFJS does this by default, but TF does not
# The TFJS imported model was having bad inference because of this
def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


# replaces the alpha channel with a white color (only 100% transparent pixels)
def replace_alpha_with_white(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        255.,
        image)

    # drops the A in RGBA
    image = image[:, :, :3]
    return image


def normalize(image):
    """
    Turns an image from the [0, 255] range into [-1, 1], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [-1, 1] range
    -------
    """
    return (image / 127.5) - 1


def denormalize(image):
    """
    Turns an image from the [-1, 1] range into [0, 255], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [0, 255] range
    -------
    """
    return (image + 1) * 127.5


# loads an image from the file system and transforms it for the network:
# (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
# puts the values in the range of [-1, 1]
def load_image(path, image_size, input_channels, output_channels, should_normalize=True):
    image = None
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=input_channels)
        image = tf.reshape(image, [image_size, image_size, input_channels])
        image = tf.cast(image, "float32")
        if input_channels == 4:
            image = blacken_transparent_pixels(image)
        if output_channels == 3:
            image = replace_alpha_with_white(image)
        if should_normalize:
            image = normalize(image)
    except UnicodeDecodeError:
        print("Error opening image in ", path)
    return image


def augment_hue_rotation(image, seed):
    image_rgb, image_alpha = image[..., 0:3], image[..., 3]
    image_rgb = tf.image.stateless_random_hue(image_rgb, 0.5, seed)
    image = tf.concat([image_rgb, image_alpha[..., tf.newaxis]], axis=-1)
    return image


def augment_translation(images):
    image = tf.concat([*images], axis=-1)
    translate = tf.keras.layers.RandomTranslation(
        (-0.15, 0.075), 0.125, fill_mode="constant", interpolation="nearest")
    image = translate(image, training=True)
    images = tf.split(image, len(images), axis=-1)
    return tf.tuple(images)


def augment(should_rotate_hue, should_translate, channels, *images):
    stacked_images = tf.stack(images, axis=0)

    # hue rotation
    if should_rotate_hue:
        hue_seed = tf.random.uniform(shape=[2], minval=0, maxval=65536, dtype="int32")
        if channels == 4:
            stacked_images_rgb, stacked_images_alpha = stacked_images[..., 0:3], stacked_images[..., 3]
            stacked_images_rgb = tf.image.stateless_random_hue(stacked_images_rgb, 0.5, hue_seed)
            stacked_images = tf.concat([stacked_images_rgb, stacked_images_alpha[..., tf.newaxis]], axis=-1)
        elif channels == 3:
            stacked_images = tf.image.stateless_random_hue(stacked_images, 0.5, hue_seed)

    # translation
    if should_translate:
        concat_channels = tf.concat(tf.unstack(stacked_images, num=len(images)), axis=-1)
        translate = tf.keras.layers.RandomTranslation(
            (-0.15 * 3, 0.075 * 3), 0.125 * 3, fill_mode="constant", interpolation="nearest")
        concat_channels = translate(concat_channels, training=True)
        stacked_images = tf.split(concat_channels, len(images), axis=-1)
        stacked_images = tf.stack(stacked_images)

    images = tf.unstack(stacked_images, num=len(images))
    return tf.tuple(images)


def normalize_all(*images):
    return tuple(map(lambda image: normalize(image), images))


def create_augmentation_with_prob(prob=0.8, should_augment_hue=True, should_augment_translation=True, channels=4):
    prob = tf.constant(prob)

    def augmentation_wrapper(*images):
        choice = tf.random.uniform(shape=[])
        inside_augmentation_probability = choice < prob
        if inside_augmentation_probability:
            return augment(should_augment_hue, should_augment_translation, channels, *images)
        else:
            return tf.tuple(images)

    return augmentation_wrapper


def create_multi_domain_image_loader(config, train_or_test_folder):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    all directions of the same character are grouped together.
    """

    domains = config.domains
    domain_folders = config.domain_folders
    data_folders = config.data_folders
    dataset_sizes = config.train_sizes if train_or_test_folder == "train" else config.test_sizes  # config.dataset_sizes
    image_size = config.image_size
    input_channels = config.input_channels
    output_channels = config.output_channels

    def load_single_image(dataset, side_index, image_number):
        path = tf.strings.join(
            [dataset, train_or_test_folder, tf.gather(domain_folders, side_index), image_number + ".png"], os.sep)
        image = load_image(path, image_size, input_channels, output_channels, False)
        return image

    @tf.function
    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        # gets the string pointing to the correct images
        dataset = tf.gather(data_folders, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # loads all images and return them
        images = [load_single_image(dataset, i, image_number) for i in range(len(domains))]
        return tuple(images)

    return load_images


def load_multi_domain_ds(config):
    should_augment_hue = not config.no_hue
    should_augment_translation = not config.no_tran
    train_size = config.train_size
    test_size = config.test_size
    batch_size = config.batch
    channels = config.inner_channels

    train_ds = tf.data.Dataset.range(train_size).shuffle(train_size)
    test_ds = tf.data.Dataset.range(test_size)

    train_ds = train_ds \
        .map(create_multi_domain_image_loader(config, "train"),
             num_parallel_calls=tf.data.AUTOTUNE)

    should_augment = should_augment_hue or should_augment_translation
    if should_augment:
        train_ds = train_ds \
            .map(create_augmentation_with_prob(0.8, should_augment_hue, should_augment_translation, channels),
                 num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds \
        .map(normalize_all, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)

    test_ds = test_ds.map(create_multi_domain_image_loader(config, "test"),
                          num_parallel_calls=tf.data.AUTOTUNE) \
        .map(normalize_all, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size)
    return train_ds, test_ds


def create_input_dropout_index_list(inputs_to_drop, number_of_domains):
    """
    Creates a list shape=(DOMAINS, TO_DROP, ? DOMAINS) that is, per possible target pose index (first dimension),
    for each possible number of dropped inputs (second dimension): all permutations of a boolean array that
    (a) nullifies the target index and (b) nullifies a number of additional inputs equal to 0, 1 or 2 (determined
    by inputs_to_drop).
    Parameters
    ----------
    inputs_to_drop a list of the number of inputs we want to drop. Must be at least [1], but can be [1, 2],
    [1, 2, 3], or [1, 3].

    Returns a 4d array with all the permutations described.
    -------

    """
    null_lists_per_target_index = []
    for target_index in range(number_of_domains):
        null_list_for_current_target = []
        for number_of_inputs_to_drop in inputs_to_drop:
            tmp_a = []
            if number_of_inputs_to_drop == 1:
                tmp = [bX == target_index for bX in range(number_of_domains)]
                tmp_a.append(tmp)

            elif number_of_inputs_to_drop == 2:
                for i_in in range(number_of_domains):
                    if not i_in == target_index:
                        tmp = [bX in [i_in, target_index] for bX in range(number_of_domains)]
                        tmp_a.append(tmp)

            elif number_of_inputs_to_drop == 3:
                for i_in in range(number_of_domains):
                    if not (i_in == target_index):
                        tmp = [(bX == target_index or (not bX == i_in)) for bX in range(number_of_domains)]
                        tmp_a.append(tmp)

            null_list_for_current_target.append(tmp_a)
        null_lists_per_target_index.append(null_list_for_current_target)

    return null_lists_per_target_index
