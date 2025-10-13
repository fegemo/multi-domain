import os
import sys
import logging
from configuration import OptionParser
import tensorflow as tf

# instructs matplotlib to use a tmp folder that is outside the network storage on verlab
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

config, parser = OptionParser().parse(sys.argv[1:], True)
logging.info(f"Running with options: {OptionParser.get_description(config, ', ', ':')}")

# configures GPU VRAM usage according to config.vram (limit, default behavior or allow growth on demand)
gpus = tf.config.list_physical_devices("GPU")
requested_gpu = config.gpu
if gpus:
    tf.config.set_visible_devices(gpus[requested_gpu], "GPU")
    if config.vram == -1:
        tf.config.experimental.set_memory_growth(gpus[requested_gpu], True)
    elif config.vram == 0:
        # do nothing -- allow tf to allocate as much as it wants at once
        pass
    else:
        # put a hard limit on the VRAM usage
        tf.config.set_logical_device_configuration(
            gpus[requested_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=config.vram)]
        )

import setup
from utility.dataset_utils import load_multi_domain_ds
from models.colla_model import CollaGANModel, CollaGANModelShuffledBatches
from models.munit_model import MunitModel
from models.remic_model import RemicModel
from models.yamata_model import YamataModel
from models.star_model import UnpairedStarGANModel, PairedStarGANModel
from models.sprite_model import SpriteEditorModel

if config.verbose:
    logging.debug(f"Tensorflow version: {tf.__version__}")

# if tf.test.gpu_device_name():
#     logging.info("Default GPU: {}".format(tf.test.gpu_device_name()))
# else:
#     logging.warning("Not using a GPU - it will take long!!")

# check if datasets need unzipping
if config.verbose:
    logging.info(f"Datasets used: {config.datasets_used}")
setup.ensure_datasets(config.verbose)

# setting the seed
if config.verbose:
    logging.debug(f"SEED set to: {config.seed}")
tf.random.set_seed(config.seed)

# loading the dataset according to the required model
train_ds, test_ds = load_multi_domain_ds(config)


# instantiates the proper model
if config.model == "stargan-unpaired":
    class_name = UnpairedStarGANModel
elif config.model == "stargan-paired":
    class_name = PairedStarGANModel
elif config.model == "collagan":
    if not config.shuffled_batches:
        class_name = CollaGANModel
    else:
        class_name = CollaGANModelShuffledBatches
elif config.model == "munit":
    class_name = MunitModel
elif config.model == "remic":
    class_name = RemicModel
elif config.model == "yamata":
    class_name = YamataModel
elif config.model == "sprite":
    class_name = SpriteEditorModel
else:
    raise Exception(f"The asked model of {config.model} was not found.")

model = class_name(config)

model.save_model_description(model.get_output_folder())
if config.verbose:
    if hasattr(model, "discriminator"):
        model.discriminator.summary()
    if hasattr(model, "generator"):
        model.generator.summary()
parser.save_configuration(model.get_output_folder(), sys.argv)


# configuration for training
steps = config.steps
epochs = config.epochs
evaluate_steps = config.evaluate_steps

logging.info(
    f"Starting training for {epochs:.2f} epochs in {steps} steps, updating visualization every "
    f"{evaluate_steps} steps...")

# starting training
callbacks = [c[len("callback_"):] for c in ["callback_debug_discriminator", "callback_evaluate_fid",
                                            "callback_evaluate_l1", "callback_early_stop"] if
             getattr(config, c)]

# tf.keras.utils.plot_model(model.generator, to_file=model.get_output_folder() + "/generator.png",
#                           show_shapes=True, show_layer_names=True, show_layer_activations=True)
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

# Sample commands (most might not work anymore, as some argument names might have changed):
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
# python train.py sprite --rm2k --steps 100 --evaluate-steps 100 --vram 4096 --temperature 0.1 --annealing linear
