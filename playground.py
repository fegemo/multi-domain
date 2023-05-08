import tensorflow as tf
from matplotlib import pyplot as plt

from dataset_utils import load_multi_domain_ds

train_ds, test_ds = load_multi_domain_ds(
    domains=["back", "left", "front", "right"],
    should_augment_translation=False,
    should_augment_hue=True)


def show_examples(batch):
    domain_images = batch
    number_of_domains = len(domain_images)
    number_of_examples = len(domain_images[0])

    plt.figure(figsize=(2 * number_of_domains, 2 * number_of_examples))
    for i in range(number_of_examples):
        for j in range(number_of_domains):
            index = i * number_of_domains + j + 1
            domain_image = domain_images[j][i]
            plt.subplot(number_of_examples, number_of_domains, index)
            plt.imshow((domain_image + 1.) / 2.)
            plt.axis("off")
    plt.show()


show_examples(next(iter(train_ds.take(1))))
