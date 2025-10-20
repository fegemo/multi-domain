import os
import sys
import logging
from configuration import OptionParser
import tensorflow as tf


os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


config, parser = OptionParser().parse(sys.argv[1:], True)
logging.info(f"Running with options: {OptionParser.get_description(config, ', ', ':')}")

from utility.dataset_utils import load_multi_domain_ds
from models.colla_model import CollaGANModel, CollaGANModelShuffledBatches
from models.munit_model import MunitModel
from models.remic_model import RemicModel
from models.yamata_model import YamataModel
from models.star_model import UnpairedStarGANModel, PairedStarGANModel
from models.sprite_model import SpriteEditorModel

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

step = model.restore_best_generator()
logging.info(f"Restored the BEST generator, which was in step {step}.")

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

