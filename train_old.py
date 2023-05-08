import sys
from math import ceil

import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_utils import load_multi_domain_ds
# from configuration import OptionParser
from star_model import UnpairedStarGANModel

from dataset_utils_old import create_unpaired_image_loader
from star_model_old import StarGANModel
from keras_utils import ConstantThenLinearDecay

DATASET_MASK = [0, 1, 0, 0, 0]
DATASET_SIZES = [912, 216, 294, 408, 12372]
DATASET_SIZES = [n * m for n, m in zip(DATASET_SIZES, DATASET_MASK)]

DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i]
              for i, n in enumerate(DATASET_SIZES)]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4
NUMBER_OF_DOMAINS = 4

train_ds = tf.data.Dataset.range(TRAIN_SIZE * NUMBER_OF_DOMAINS).shuffle(TRAIN_SIZE * NUMBER_OF_DOMAINS)
train_ds = train_ds.map(create_unpaired_image_loader(TRAIN_SIZES, "train", True), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)

test_ds = tf.data.Dataset.range(TEST_SIZE * NUMBER_OF_DOMAINS).shuffle(TEST_SIZE * NUMBER_OF_DOMAINS)
test_ds = test_ds.map(create_unpaired_image_loader(TEST_SIZES, "test", True))
test_ds = test_ds.batch(BATCH_SIZE)

model = StarGANModel(train_ds, test_ds,
                     "rm2k-240ep-noaug-lrdecay-lr0.0001-tfadd0.16.1-discwithdecay-old", "stargan-unpaired",
                     generator_type="stargan",
                     discriminator_steps=5,
                     lambda_domain=1.,
                     lambda_reconstruction=10.,
                     lambda_gp=10.)

EPOCHS = 240
STEPS = ceil(TRAIN_SIZE / BATCH_SIZE) * EPOCHS
UPDATE_STEPS = STEPS // 40

model.generator_optimizer = tf.keras.optimizers.Adam(ConstantThenLinearDecay(0.0001, STEPS//model.discriminator_steps), beta_1=0.5, beta_2=0.999)
model.discriminator_optimizer = tf.keras.optimizers.Adam(ConstantThenLinearDecay(0.0001, STEPS), beta_1=0.5, beta_2=0.999)

print("Starting training...")
model.fit(STEPS, UPDATE_STEPS, callbacks=[])
