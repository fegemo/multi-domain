import tensorflow as tf
import os

DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
    for folder
    in DATASET_NAMES
]

DOMAINS = ["back", "left", "front", "right"]
DOMAIN_FOLDERS = [f"{i}-{name}" for i, name in enumerate(DOMAINS)]
INPUT_CHANNELS = 4
IMG_SIZE = 64
OUTPUT_CHANNELS = 4


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


def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, INPUT_CHANNELS)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


def load_image(path, should_normalize=True):
    image = None
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=INPUT_CHANNELS)
        image = tf.reshape(image, (IMG_SIZE, IMG_SIZE, INPUT_CHANNELS))
        image = tf.cast(image, "float32")
        image = blacken_transparent_pixels(image)
        if should_normalize:
            image = normalize(image)
    except UnicodeDecodeError:
        print("Error opening image in ", path)
    return image


def create_unpaired_image_loader(dataset_sizes, train_or_test_folder, should_normalize):
    """
    Creates an image loader for the datasets (as configured in configuration.py) in such a way that
    images are all unrelated but keep a label of which side it is from.
    They are unrelated because, e.g., the front and right sides of a sprite are not paired.
    Used for unpaired (unsupervised) learning such as StarGAN.
    """

    @tf.function
    def load_images(image_number):
        image_number = tf.cast(image_number, "int32")

        dataset_index = tf.constant(0, dtype="int32")
        condition = lambda which_image, which_dataset: which_image >= tf.gather(dataset_sizes, which_dataset) * 4
        body = lambda which_image, which_dataset: [which_image - tf.gather(dataset_sizes, which_dataset) * 4,
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])
        file_number = image_number // 4
        side_index = image_number % 4

        # defines the angle to which rotate hue
        hue_angle = tf.random.uniform(shape=[], minval=-1., maxval=1.)

        # gets the string pointing to the correct images
        dataset = tf.gather(DATA_FOLDERS, dataset_index)
        file_number = tf.strings.as_string(file_number)
        side_folder = tf.gather(DOMAIN_FOLDERS, side_index)

        # loads and transforms the images according to how the generator and discriminator expect them to be
        image_path = tf.strings.join([dataset, train_or_test_folder, side_folder, file_number + ".png"], os.sep)
        image = load_image(image_path, should_normalize)
        label = tf.one_hot(side_index, len(DOMAIN_FOLDERS))

        return image, label

    return load_images
