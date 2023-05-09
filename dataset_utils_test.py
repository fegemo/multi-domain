from unittest.mock import patch
import tensorflow as tf

from dataset_utils import load_image


class TestDatasetUtils(tf.test.TestCase):
    @patch("dataset_utils.tf.io.read_file")
    @patch("dataset_utils.tf.image.decode_png")
    def test_load_image(self, mock_tf_decode_png, mock_tf_read_file):
        sample_decoded_png = tf.constant([
            [
                [0, 127, 255],
                [0, 127, 255]],
            [
                [0, 127, 255],
                [0, 127, 255]
             ]
        ])
        # mock_tf_read_file.return_value = None
        mock_tf_decode_png.return_value = sample_decoded_png
        loaded = load_image("any-path-as-it-was-mocked.png", 2, 3, should_normalize=True)

        mock_tf_read_file.assert_called()
        mock_tf_decode_png.assert_called()

        # shape should be (2, 2, 3)
        self.assertAllEqual(tf.shape(loaded), tf.constant([2, 2, 3]))

        # type should be float32
        self.assertDTypeEqual(loaded, "float32")

        # intensities should be in [-1, 1]
        self.assertAllInRange(loaded, -1., 1.)

        # intensities as: -1 for red, close to 0 for green, +1 for blue
        self.assertAllClose(loaded, tf.constant([
            [
                [-1., 0., 1.],
                [-1., 0., 1.]
            ],
            [
                [-1., 0., 1.],
                [-1., 0., 1.]
            ]
        ]), atol=0.01)

    def setUp(self):
        super(TestDatasetUtils, self).setUp()

    def tearDown(self):
        pass


if __name__ == '__main__':
    tf.test.main()
