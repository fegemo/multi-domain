import logging
import sys
from math import ceil

import tensorflow as tf

# allows tf to use all the amount of vram of the device
# important for running on low vram environments (as my local 4gb)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )

from colla_model import CollaGANModel
from dataset_utils import load_multi_domain_ds
from configuration import OptionParser
from munit_model import MunitModel
from remic_model import RemicModel
from star_model import UnpairedStarGANModel, PairedStarGANModel
import setup

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


config, parser = OptionParser().parse(sys.argv[1:], True)
logging.info(f"Running with options: {parser.get_description(', ', ':')}")
if config.verbose:
    logging.debug(f"Tensorflow version: {tf.__version__}")

if tf.test.gpu_device_name():
    logging.info("Default GPU: {}".format(tf.test.gpu_device_name()))
else:
    logging.warning("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if config.verbose:
    logging.info("Datasets used: ", config.datasets_used)
setup.ensure_datasets(config.verbose)

# setting the seed
if config.verbose:
    logging.debug("SEED set to: ", config.seed)
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
elif config.model == "collagan":
    class_name = CollaGANModel
elif config.model == "munit":
    class_name = MunitModel
elif config.model == "remic":
    class_name = RemicModel
else:
    raise Exception(f"The asked model of {config.model} was not found.")

model = class_name(config)

model.save_model_description(model.get_output_folder())
if config.verbose:
    model.discriminator.summary()
    model.generator.summary()
parser.save_configuration(model.get_output_folder(), sys.argv)


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
steps = config.steps
epochs = steps / ceil(len(train_ds) / config.batch)
evaluate_steps = config.evaluate_steps

logging.info(
    f"Starting training for {epochs:.2f} epochs in {steps} steps, updating visualization every "
    f"{evaluate_steps} steps...")

# starting training
callbacks = [c[len("callback_"):] for c in ["callback_debug_discriminator", "callback_evaluate_fid",
                                            "callback_evaluate_l1"] if
             getattr(config, c)]


model.fit(train_ds, test_ds, steps, evaluate_steps, callbacks=callbacks)

# restores the best generator (best l1 - priority, or best fid)
step = model.restore_best_generator()
logging.info(f"Restored the BEST generator, which was in step {step}.")

if config.save_model:
    logging.info(f"Saving the generator...")
    model.save_generator()
    logging.info(f"Generator saved.")

# generating resulting images
num_examples_to_generate = tf.minimum(100, config.test_size)
skip_examples = tf.maximum(1, config.test_size // num_examples_to_generate)
skip_examples = tf.cast(skip_examples, tf.int64)
num_examples_to_generate = tf.cast(num_examples_to_generate, tf.int64)
logging.info(f"Starting to generate {num_examples_to_generate} images from the test dataset "
             f"with generator from step {step}, "
             f"hopping {skip_examples} example(s) each time.")
skipping_test_ds = (
    test_ds.unbatch()
    .enumerate()
    .filter(lambda c, _: c % skip_examples == 0)
    .take(num_examples_to_generate)
)
model.generate_images_from_dataset(skipping_test_ds, step, num_images=num_examples_to_generate.numpy())


logging.info("Finished executing.")

# python train.py stargan-unpaired --rm2k --log-folder output --epochs 240 --no-aug --model-name stargan-unpaired --experiment rm2k-240ep-noaug-lrdecay-lr0.0001-tfadd0.18.0-discwithdecay-bce-singletargetdomain --sampler single-target
# python train.py stargan-paired --rm2k --log-folder output --epochs 4 --no-aug --model-name playground --experiment playground
# python train.py collagan --rm2k --log-folder output --epochs 40 --no-aug --model-name collagan --experiment playground
# python train.py collagan --rm2k --log-folder output --epochs 40 --no-aug --model-name collagan --experiment b4,l1100,lr0.0002,dom10 --batch 4 --lambda-l1 100 --lr 0.0002 --lambda-domain 10
# python train.py collagan --rm2k --log-folder output --epochs 300 --no-aug --model-name collagan --experiment playground --lambda-l1 100 --lr 0.0002 --lr-decay none --callback-evaluate-l1 --callback-evaluate-fid --callback-debug-discriminator
# python train.py collagan --rm2k --log-folder output --epochs 400 --no-tran --model-name collagan --experiment l1100,d10,s10,lr0001,decayctl,indrop
# python train.py collagan --rm2k --log-folder output --epochs 400 --no-tran --model-name collagan --experiment l1100,d10,s10,lr0001,decayctl,indrop,correctssim,correctdomain --lambda-l1 100 --lambda-domain 10 --lambda-ssim 10 --lr 0.0001 --lr-decay constant-than-linear --callback-evaluate-l1 --callback-evaluate-fid --callback-debug-discriminator --input-dropout
# python train.py munit --log-folder output --steps 4000 --evaluate-steps 1000 --lr 0.0001 --batch 1 --lr-decay step --model-name munit --experiment all,lambda-l1-1,lambda-latent-reconstruction-0 --lambda-l1 1 --lambda-latent-reconstruction 0 --callback-evaluate-fid --callback-evaluate-l1 --save-model --tiny --rm2k --rmxp --rmvx --misc --no-tran
# python train.py munit --log-folder output --steps 4000 --evaluate-steps 250 --lr 0.0001 --batch 1 --lr-decay step --model-name munit --experiment pytorch-impl --lambda-l1 10 --lambda-latent-reconstruction 1 --lambda-cyclic-reconstruction 10 --domains back left --callback-evaluate-fid --callback-evaluate-l1 --rmxp
# python train.py remic --steps 4000 --evaluate-steps 500 --lr 0.0001 --batch 3 --model-name remic --rmxp --lambda-l1 10 --lambda-latent-reconstruction 1 --lambda-cyclic-reconstruction 20 --discriminator-scales 1 --lr-decay none
