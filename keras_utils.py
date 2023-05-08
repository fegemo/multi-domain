import tensorflow as tf


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
