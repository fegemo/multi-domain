import sys
from math import ceil

import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_utils import load_multi_domain_ds
from configuration import OptionParser
from star_model import UnpairedStarGANModel, PairedStarGANModel
import setup

config, parser = OptionParser().parse(sys.argv[1:], True)
print("Running with options:", parser.get_description(", ", ":"))
if config.verbose:
    print("Running with options: ", config)
    print("Tensorflow version: ", tf.__version__)

if tf.test.gpu_device_name():
    print("Default GPU: {}".format(tf.test.gpu_device_name()))
else:
    print("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if config.verbose:
    print("Datasets used: ", config.datasets_used)
setup.ensure_datasets(config.verbose)

# setting the seed
if config.verbose:
    print("SEED set to: ", config.seed)
tf.random.set_seed(config.seed)

# loading the dataset according to the required model
train_ds, test_ds = load_multi_domain_ds(config)

# previews one batch of images from the test dataset
# sample_batch = next(iter(test_ds))
# batch_shape = tf.shape(sample_batch).numpy()
# number_of_domains, batch_size = batch_shape[0], batch_shape[1]
# fig = plt.figure(figsize=(4*number_of_domains, 4*batch_size))
# for i in range(batch_size):
#     idx = i * number_of_domains + 1
#     for j in range(number_of_domains):
#         plt.subplot(batch_size, number_of_domains, idx)
#         plt.imshow(sample_batch[j][i] * 0.5 + 0.5)
#         plt.axis("off")
#
#         idx += 1
# plt.show()
# plt.close()


# instantiates the proper model
if config.model == "stargan-unpaired":
    class_name = UnpairedStarGANModel
elif config.model == "stargan-paired":
    class_name = PairedStarGANModel
else:
    raise Exception(f"The asked model of {config.model} was not found.")

model = class_name(config)

model.save_model_description(model.get_output_folder())
if config.verbose:
    model.discriminator.summary()
    model.generator.summary()
parser.save_configuration(model.get_output_folder())


# batch = next(iter(train_ds))
# random_input = model.select_random_input(batch, options.batch)
# image_and_domain, random_source_index, random_source_image, random_target_index = random_input
# random_target_index = tf.argmax(random_target_index)
# target_images = tf.gather(tf.transpose(batch, [1, 0, 2, 3, 4]), random_target_index, axis=1, batch_dims=1)
#
#
# for i in range(options.batch):
#     s = random_source_index[i]
#     t = random_target_index[i]
#     print(f"input {i} of batch has source {s} and target {t}")
#
#     plt.figure(figsize=(4*2, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(random_source_image[i] * 0.5 + 0.5)
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     plt.imshow(target_images[i] * 0.5 + 0.5)
#     plt.axis("off")
# plt.show()

# configuration for training
steps = ceil(config.train_size / config.batch) * config.epochs
update_steps = steps // 40

print(
    f"Starting training for {config.epochs} epochs in {steps} steps, updating visualization every "
    f"{update_steps} steps...")

# starting training
callbacks = [c[len("callback_"):] for c in ["callback_debug_discriminator", "callback_evaluate_fid",
                                            "callback_evaluate_l1"] if
             getattr(config, c)]

# tf.random.set_seed(11)
# image_and_domain, random_source_index, random_source_image, random_target_index = model.select_random_input(next(iter(train_ds)), 4)
# print("random_source_index", random_source_index)
# print("random_target_index", random_target_index)


model.fit(train_ds, test_ds, steps, update_steps, callbacks=callbacks)

# restores the best generator (best l1 - priority, or best fid)
model.restore_best_generator()

# generating resulting images
model.generate_images_from_dataset(test_ds)

if config.save_model:
    print(f"Saving the generator...")
    model.save_generator()

print("Finished executing.")

# python train.py stargan-unpaired --rm2k --log-folder output --epochs 240 --no-aug --model-name stargan-unpaired --experiment rm2k-240ep-noaug-lrdecay-lr0.0001-tfadd0.18.0-discwithdecay-bce-singletargetdomain --sampler single-target
# python train.py stargan-paired --rm2k --log-folder output --epochs 4 --no-aug --model-name playground --experiment playground
