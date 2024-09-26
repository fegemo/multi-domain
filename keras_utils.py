import tensorflow as tf
from tensorflow import keras
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
    return tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in network.trainable_weights])