import tensorflow as tf

NUMBER_OF_DOMAINS = 4
IMG_SIZE = 64


def random_domain_label(batch_size):
    random_domain = tf.random.uniform(shape=[batch_size, ], minval=0, maxval=NUMBER_OF_DOMAINS, dtype="int32")
    onehot_domain_label = tf.one_hot(random_domain, NUMBER_OF_DOMAINS)
    return onehot_domain_label


def channelize_domain_label(onehot_domain_label):
    channelized_domain_label = onehot_domain_label[:, tf.newaxis, tf.newaxis, :]
    channelized_domain_label = tf.tile(channelized_domain_label, [1, IMG_SIZE, IMG_SIZE, 1])
    return channelized_domain_label


def concat_image_and_domain_label(image, onehot_label):
    channelized_domain_label = channelize_domain_label(onehot_label)
    return tf.concat([image, channelized_domain_label], axis=-1)
