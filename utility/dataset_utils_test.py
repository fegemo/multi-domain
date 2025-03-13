from unittest.mock import patch
import tensorflow as tf

from configuration import OptionParser
from .dataset_utils import load_image, create_multi_domain_image_loader, blacken_transparent_pixels


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
        loaded = load_image("any-path-as-it-was-mocked.png", 2, 3, 3, should_normalize=True)

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


class TestMultiImageLoader(tf.test.TestCase):
    @patch("dataset_utils.load_image")
    def test_multiple_datasets(self, mock_load_image):
        mock_load_image.side_effect = lambda path, i, c, norm: path
        # mock_load_image.return_value = lambda path, i, c, norm: path

        config = OptionParser().parse(["stargan-paired", "--rm2k", "--tiny"])
        loader = create_multi_domain_image_loader(config, "train")
        last_tiny = loader(775)
        first_rm2k = loader(776)

        self.assertIn("tiny-hero", last_tiny[0].numpy().decode("utf-8"))
        self.assertIn("rpg-maker-2000", first_rm2k[0].numpy().decode("utf-8"))

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()


class TestBlackenTransparentPixels(tf.test.TestCase):
    def test_blacken_transparent_pixels(self):
        image_with_transparency = tf.constant([
            [[1., 1., 1., 0.], [0., 0., 0., 0.]],
            [[.4, .4, .4, .1], [0., 0., 0., 1.]]
        ])
        expected_blackened = tf.constant([
            [[0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[.4, .4, .4, .1], [0., 0., 0., 1.]]
        ])

        result = blacken_transparent_pixels(image_with_transparency)
        self.assertAllEqual(result, expected_blackened)

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    tf.test.main()
