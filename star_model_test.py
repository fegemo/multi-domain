# import unittest
import argparse
from unittest import mock
from unittest.mock import patch

import tensorflow as tf

from configuration import OptionParser
from star_model import UnpairedStarGANModel


class StarGANModelTest(tf.test.TestCase):
    def skip___test_concat_image_and_domain(self):

        # easy [2, 1, 1, 2+2] test
        batch_size = 2
        image_size = 1
        number_of_channels = 2
        number_of_domains = 2
        config = OptionParser().parse(["stargan-unpaired", "--image-size", str(image_size), "--rmxp", "--no-aug", "--batch", str(batch_size)])
        model = UnpairedStarGANModel(config)

        image = tf.constant([[[1, 0]]])
        image = image[tf.newaxis, ...]
        batch = tf.concat([image, image*5], axis=0)

        domains_oh = tf.constant([[0, 1], [1, 0]])
        image_and_domain = model.sampler.concat_image_and_domain(batch, domains_oh)
        self.assertShapeEqual(
            tf.ones([batch_size, image_size, image_size, number_of_channels + number_of_domains]),
            image_and_domain)
        self.assertAllEqual(tf.constant([[[[1, 0, 0, 1]]], [[[5, 0, 1, 0]]]]), image_and_domain)

        # [3, 2, 2, 4+4] test
        batch_size = 3
        image_size = 2
        number_of_channels = 4
        number_of_domains = 4
        config = OptionParser().parse(["stargan-unpaired", "--image-size", str(image_size), "--rmxp", "--no-aug", "--batch", str(batch_size)])
        model = UnpairedStarGANModel(config)

        image = tf.constant([[[1, 1, 1, 0], [2, 2, 2, 0]], [[3, 3, 3, 0], [4, 4, 4, 0]]])
        image = image[tf.newaxis, ...]
        batch = tf.concat([image, image * 10, image*100], axis=0)

        domains = tf.constant([3, 1, 1])
        domains_oh = tf.constant([[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]])

        image_and_domain = model.sampler.concat_image_and_domain(batch, domains_oh)
        self.assertShapeEqual(
            tf.ones([batch_size, image_size, image_size, number_of_channels + number_of_domains]),
            image_and_domain)
        self.assertAllEqual(
            tf.constant([
                [[[1, 1, 1, 0, 0, 0, 0, 1], [2, 2, 2, 0, 0, 0, 0, 1]], [[3, 3, 3, 0, 0, 0, 0, 1], [4, 4, 4, 0, 0, 0, 0, 1]]],
                [[[10, 10, 10, 0, 0, 1, 0, 0], [20, 20, 20, 0, 0, 1, 0, 0]], [[30, 30, 30, 0, 0, 1, 0, 0], [40, 40, 40, 0, 0, 1, 0, 0]]],
                [[[100, 100, 100, 0, 0, 1, 0, 0], [200, 200, 200, 0, 0, 1, 0, 0]], [[300, 300, 300, 0, 0, 1, 0, 0], [400, 400, 400, 0, 0, 1, 0, 0]]]
            ]),
            image_and_domain
        )

    def setup(self):
        super(StarGANModelTest, self).setUp()

    def tearDown(self):
        pass


if __name__ == '__main__':
    tf.test.main()
