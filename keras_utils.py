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


class AdaptiveMixing(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reshape = None
        self.dense_scale = None
        self.dense_bias = None
        self.x_channels = None

    def build(self, input_shapes):
        self.x_channels = input_shapes[0][-1]
        self.dense_scale = keras.layers.Dense(self.x_channels)
        self.dense_bias = keras.layers.Dense(self.x_channels)
        self.reshape = keras.layers.Reshape([1, 1, -1])
        
    def call(self, inputs, **kwargs):
        x, w = inputs
        ys = self.dense_scale(w)
        yb = self.dense_bias(w)
        ys = self.reshape(ys)
        yb = self.reshape(yb)

        return ys * x + yb



# import tensorflow as tf
# import keras_utils as ku
#
# image_input = tf.keras.layers.Input(shape=[2, 2, 3])
# palet_input = tf.keras.layers.Input(shape=[64, 3])
# x = image_input
# w = palet_input
# w = tf.keras.layers.Flatten()(w)
# w = tf.keras.layers.Dense(10)(w)
# output = ku.AdaptiveMixing()([x, w])
# model = tf.keras.models.Model(inputs=[image_input, palet_input], outputs=[output])
#
# example_image = tf.expand_dims(tf.constant([[[0, 0, 0], [0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6], [1., 1., 1.]]], dtype=tf.float32), 0)
# # example_palet = tf.RaggedTensor.from_row_splits(values=[[.5, .5, .5], [.2, .2, .2]], row_splits=[0, 2])
# example_palet = tf.expand_dims(tf.concat([tf.constant([[.5, .5, .5,], [.2, .2, .2]]), tf.ones([62, 3]) * 1000.], axis=0), 0)
# model([example_image, example_palet])