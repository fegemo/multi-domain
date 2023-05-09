import tensorflow as tf
from tensorflow import keras


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
