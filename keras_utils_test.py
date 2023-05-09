import tensorflow as tf

from keras_utils import TileLayer


class TestTileLayer(tf.test.TestCase):
    def test_call(self):
        # shape=(2, 4) that should become (2, 3, 4) then (2, 3, 3, 4)
        # shape=(b, d) that should become (b, s, d) then (2, s, s, d)
        batch = tf.constant([[0, 1, 0, 0], [1, 0, 0, 0]])
        tile_size = 3

        layer = TileLayer(tile_size)
        result = layer(batch)

        # shape should have one rank higher, in the middle
        self.assertAllEqual(tf.shape(result), [2, 3, 4])
        # shape should respect the tile size
        self.assertEqual(tf.shape(result)[1], tile_size)

        # values should have been repeated
        self.assertAllEqual(result, tf.constant([
            [   # element 0 of batch
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0]
            ],
            [   # element 1 of batch
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ]
        ]))

        # shape should now have even one more rank
        result = layer(result)
        self.assertAllEqual(tf.shape(result), [2, 3, 3, 4])


if __name__ == '__main__':
    tf.test.main()
