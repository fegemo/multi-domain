import argparse
import os
import datetime
import sys
from math import ceil

SEED = 42

DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
    for folder
    in DATASET_NAMES
]

DOMAINS = ["back", "left", "front", "right"]
DOMAIN_BACK = DOMAINS.index("back")  # 0
DOMAIN_LEFT = DOMAINS.index("left")  # 1
DOMAIN_FRONT = DOMAINS.index("front")  # 2
DOMAIN_RIGHT = DOMAINS.index("right")  # 3
# ["0-back", "1-left", "2-front", "3-right"]
DOMAIN_FOLDERS = [f"{i}-{name}" for i, name in enumerate(DOMAINS)]

DATASET_MASK = [1, 1, 1, 1, 1]
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

LAMBDA_GP = 10.
LAMBDA_DOMAIN = 1.
LAMBDA_RECONSTRUCTION = 10.
DISCRIMINATOR_STEPS = 5
EPOCHS = 160
LR_DECAY = "constant-then-linear"
LR = 0.0001
TRAINING_SAMPLER = "multi-target"

LOG_FOLDER = "temp-side2side"


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptionParser(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.values = {}

    def initialize(self):
        self.parser.add_argument(
            "model", help="one from { stargan-unpaired, stargan-paired, collagan-unpaired, collagan-paired }"
                          "- the model to train")
        self.parser.add_argument("--generator", help="network from { resnet, unet } for stargan or "
                                                     "{ XXXX } for collagan", default="")
        self.parser.add_argument("--discriminator", help="network from { resnet, unet } for stargan or "
                                                         "{ XXXX } for collagan", default="")
        self.parser.add_argument("--image-size", help="size of squared images", default=IMG_SIZE, type=int)
        self.parser.add_argument("--output-channels", help="size of squared images", default=OUTPUT_CHANNELS, type=int)
        self.parser.add_argument("--input-channels", help="size of squared images", default=INPUT_CHANNELS, type=int)
        self.parser.add_argument("--domains", help="domain folder names (w/o number, but in order)",
                                 default=DOMAINS, nargs="+")
        self.parser.add_argument("--verbose", help="outputs verbosity information",
                                 default=False, action="store_true")

        self.parser.add_argument("--rmxp", action="store_true", default=False, help="Uses RPG Maker XP dataset")
        self.parser.add_argument("--rm2k", action="store_true", default=False, help="Uses RPG Maker 2000"
                                                                                    " dataset")
        self.parser.add_argument("--rmvx", action="store_true", default=False, help="Uses RPG Maker VX Ace"
                                                                                    " dataset")
        self.parser.add_argument("--tiny", action="store_true", default=False, help="Uses the Tiny Hero dataset")
        self.parser.add_argument("--misc", action="store_true", default=False, help="Uses the miscellaneous"
                                                                                    " sprites dataset")
        self.parser.add_argument("--rmxp-validation", action="store_true", default=False, help="Uses only RMXP (44 "
                                                                                               "test examples) for "
                                                                                               "validation and to "
                                                                                               "generate images in "
                                                                                               "the end")
        self.parser.add_argument("--rm2k-validation", action="store_true", default=False, help="Uses only RM2K (32 "
                                                                                               "test examples) for "
                                                                                               "validation and to "
                                                                                               "generate images in "
                                                                                               "the end")
        self.parser.add_argument("--rmvx-validation", action="store_true", default=False, help="Uses only RMVX (61 "
                                                                                               "test examples) for "
                                                                                               "validation and to "
                                                                                               "generate images in "
                                                                                               "the end")
        self.parser.add_argument("--tiny-validation", action="store_true", default=False, help="Uses only tiny (136 "
                                                                                               "test examples) for "
                                                                                               "validation and to "
                                                                                               "generate images in "
                                                                                               "the end")

        self.parser.add_argument("--batch", type=int, help="the batch size", default=BATCH_SIZE)
        self.parser.add_argument("--lr-decay", help="one from {none, constant-then-linear}", default=LR_DECAY)
        self.parser.add_argument("--lr", type=float, help="(initial) learning rate", default=LR)
        self.parser.add_argument(
            "--lambda-gp", type=float, help="value for λgradient_penalty used in stargan", default=LAMBDA_GP)
        self.parser.add_argument("--lambda-domain", type=float,
                                 help="value for λdomain used in stargan", default=LAMBDA_DOMAIN)
        self.parser.add_argument("--lambda-reconstruction", type=float,
                                 help="value for λreconstruction used in stargan", default=LAMBDA_RECONSTRUCTION)
        self.parser.add_argument("--d-steps", type=float,
                                 help="number of discriminator updates for each generator in stargan",
                                 default=DISCRIMINATOR_STEPS)
        self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=EPOCHS)
        self.parser.add_argument("--no-aug", action="store_true", help="Disables all augmentation", default=False)
        self.parser.add_argument("--no-hue", action="store_true", help="Disables hue augmentation", default=False)
        self.parser.add_argument("--no-tran", action="store_true", help="Disables translation augmentation",
                                 default=False)
        self.parser.add_argument("--sampler", help="one from {multi-target, single-target} indicating whether batches "
                                                   "are trained with the same target (single) or with each sample"
                                                   "having its own (multi)", default=TRAINING_SAMPLER)

        self.parser.add_argument("--callback-debug-discriminator",
                                 help="every few update steps, show the discriminator output with some images from "
                                      "the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--callback-evaluate-fid",
                                 help="every few update steps, evaluate with the FID metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--callback-evaluate-l1",
                                 help="every few update steps, evaluate with the L1 metric the performance "
                                      "on the train and test sets",
                                 default=False, action="store_true")
        self.parser.add_argument("--save-model", help="saves the model at the end", default=False, action="store_true")
        self.parser.add_argument("--keep-checkpoint", help="checkpoints training", default=False, action="store_true")
        self.parser.add_argument("--model-name", help="architecture name", default="some-architecture")
        self.parser.add_argument("--experiment", help="description of this experiment", default="playground")
        self.parser.add_argument(
            "--log-folder", help="the folder in which the training procedure saves the logs", default=LOG_FOLDER)
        self.initialized = True

    def parse(self, args=None, return_parser=False):
        if args is None:
            args = sys.argv[1:]
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args(args)

        setattr(self.values, "number_of_domains", len(self.values.domains))
        setattr(self.values, "seed", SEED)
        if self.values.no_aug:
            setattr(self.values, "no_hue", True)
            setattr(self.values, "no_tran", True)
        datasets_used = list(filter(lambda opt: getattr(self.values, opt), ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        setattr(self.values, "datasets_used", datasets_used)
        if len(datasets_used) == 0:
            raise Exception("No dataset was supplied with: --tiny, --rm2k, --rmxp, --rmvx, --misc")
        setattr(self.values, "run_string", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        global DATASET_MASK
        global DATASET_SIZES
        global DATASET_SIZE
        global TRAIN_SIZES
        global TEST_SIZES
        global TRAIN_SIZE
        global TEST_SIZE
        global BUFFER_SIZE

        DATASET_MASK = list(
            map(lambda opt: 1 if getattr(self.values, opt) else 0, ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        DATASET_SIZES = [912, 216, 294, 408, 12372]
        DATASET_SIZES = [n * m for n, m in zip(DATASET_SIZES, DATASET_MASK)]
        DATASET_SIZE = sum(DATASET_SIZES)
        TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
        TRAIN_SIZE = sum(TRAIN_SIZES)
        TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i]
                      for i, n in enumerate(DATASET_SIZES)]
        if self.values.rmxp_validation:
            TEST_SIZES = [0, 0, 44, 0, 0]
        elif self.values.rm2k_validation:
            TEST_SIZES = [0, 32, 0, 0, 0]
        elif self.values.rmvx_validation:
            TEST_SIZES = [0, 0, 0, 61, 0]
        elif self.values.tiny_validation:
            TEST_SIZES = [136, 0, 0, 0, 0]

        TEST_SIZE = sum(TEST_SIZES)
        BUFFER_SIZE = DATASET_SIZE

        setattr(self.values, "train_sizes", TRAIN_SIZES)
        setattr(self.values, "train_size", TRAIN_SIZE)
        setattr(self.values, "test_sizes", TEST_SIZES)
        setattr(self.values, "test_size", TEST_SIZE)
        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path):
        from io_utils import ensure_folder_structure
        ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(self.get_description("\n", ": ") + "\n")


def in_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# if not in_notebook():
#     options = OptionParser([]).parse()
#
#     BATCH_SIZE = options.batch
#     # MAX_PALETTE_SIZE = options.max_palette_size
#     LOG_FOLDER = options.log_folder
