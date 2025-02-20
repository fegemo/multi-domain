import argparse
import os
import datetime
import sys
from math import ceil

SEED = 42

DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]
DOMAINS = ["back", "left", "front", "right"]
TRAIN_PERCENTAGE = 0.85
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 4

LAMBDA_GP = 10.
LAMBDA_DOMAIN = 1.
LAMBDA_RECONSTRUCTION = 10.
LAMBDA_L1 = 10.
LAMBDA_SSIM = 10.
LAMBDA_PALETTE = 0.
LAMBDA_TV = 0.
LAMBDA_LATENT_RECONSTRUCTION = 1.
LAMBDA_CYCLIC_RECONSTRUCTION = 0  # 10 for munit on summer<>>winter and cityscapes<>synthia datasets (MUNIT)
DISCRIMINATOR_STEPS = 5
EPOCHS = 160
PRETRAIN_EPOCHS = 0  # 30 in colla's code
LR_DECAY = "constant-then-linear"
LR = 0.0001
TRAINING_SAMPLER = "multi-target"

LOG_FOLDER = "output"


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
            "model", help="one from { stargan-unpaired, stargan-paired, collagan, yamatagan }"
                          "- the model to train")
        self.parser.add_argument("--generator", help="network from { resnet, unet } for stargan or "
                                                     "{ affluent } for collagan", default="")
        self.parser.add_argument("--discriminator", help="network from { resnet, unet } for stargan or "
                                                         "{ original } for collagan", default="")
        self.parser.add_argument("--conditional-discriminator", help="Makes the discriminator receive"
                                                                     " the source image just like in Pix2Pix (only"
                                                                     " for stargan-paired and collagan).",
                                 action="store_true", default=False)
        self.parser.add_argument("--source-domain-aware-generator", help="Makes the generator receive the source domain"
                                                                         " besides the target and source image"
                                                                         " (only for stargan-paired)",
                                 action="store_true", default=False)
        self.parser.add_argument("--image-size", help="size of squared images", default=IMG_SIZE, type=int)
        self.parser.add_argument("--output-channels", help="size of squared images", default=OUTPUT_CHANNELS, type=int)
        self.parser.add_argument("--input-channels", help="size of squared images", default=INPUT_CHANNELS, type=int)
        self.parser.add_argument("--verbose", help="outputs verbosity information",
                                 default=False, action="store_true")

        self.parser.add_argument("--batch", type=int, help="the batch size", default=BATCH_SIZE)
        self.parser.add_argument("--lr-decay", help="one from {none, constant-then-linear}", default=LR_DECAY)
        self.parser.add_argument("--lr", type=float, help="(initial) learning rate", default=LR)
        self.parser.add_argument(
            "--lambda-gp", type=float, help="value for λgradient_penalty used in stargan", default=LAMBDA_GP)
        self.parser.add_argument("--lambda-domain", type=float,
                                 help="value for λdomain used in stargan and collagan", default=LAMBDA_DOMAIN)
        self.parser.add_argument("--lambda-reconstruction", type=float,
                                 help="value for λreconstruction used in stargan", default=LAMBDA_RECONSTRUCTION)
        self.parser.add_argument("--lambda-l1", type=float, help="value for λl1 used in paired stargan,"
                                                                 " for λl1_forward in collagan, and λreconstruction"
                                                                 " in munit",
                                 default=LAMBDA_L1)
        self.parser.add_argument("--lambda-l1-backward", type=float, help="value for λl1_backward (cyclic) used "
                                                                          "in collagan")
        self.parser.add_argument("--lambda-ssim", type=float, help="value for λssim used in collagan",
                                 default=LAMBDA_SSIM)
        self.parser.add_argument("--lambda-palette", type=float, help="value for λpalette used in paired "
                                                                      "stargan and collagan",
                                 default=LAMBDA_PALETTE)
        self.parser.add_argument("--lambda-tv", type=float, help="value for λtotal-variation used in paired stargan",
                                 default=LAMBDA_TV)
        self.parser.add_argument("--lambda-latent-reconstruction", type=float,
                                 help="value for λlatent-reconstruction used in munit",
                                 default=LAMBDA_LATENT_RECONSTRUCTION)
        self.parser.add_argument("--lambda-cyclic-reconstruction", type=float,
                                 help="value for λcyclic-reconstruction used in munit",
                                 default=LAMBDA_CYCLIC_RECONSTRUCTION)
        self.parser.add_argument("--d-steps", type=int,
                                 help="number of discriminator updates for each generator in stargan",
                                 default=DISCRIMINATOR_STEPS)
        self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=EPOCHS)
        self.parser.add_argument("--steps", type=int, help="number of generator update steps to train", default=None)
        self.parser.add_argument("--evaluate-steps", type=int, help="number of generator update steps "
                                                                    "to wait until an evaluation is done", default=1000)
        self.parser.add_argument("--pretrain-epochs", type=int, help="number of epochs pretraining (used by collagan)",
                                 default=PRETRAIN_EPOCHS)
        self.parser.add_argument("--no-aug", action="store_true", help="Disables all augmentation", default=False)
        self.parser.add_argument("--no-hue", action="store_true", help="Disables hue augmentation", default=False)
        self.parser.add_argument("--no-tran", action="store_true", help="Disables translation augmentation",
                                 default=False)
        self.parser.add_argument("--sampler", help="one from {multi-target, single-target} indicating whether "
                                                   "batches are trained with the same target (single) or with "
                                                   "each sample having its own (multi)", default=TRAINING_SAMPLER)
        self.parser.add_argument("--capacity", type=int, help="capacity multiplier of CollaGAN's and "
                                                              "StarGAN's generator", default=1)
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
        self.parser.add_argument("--save-model", help="saves the model at the end", default=False,
                                 action="store_true")
        self.parser.add_argument("--model-name", help="architecture name", default="some-architecture")
        self.parser.add_argument("--experiment", help="description of this experiment", default="playground")
        self.parser.add_argument(
            "--log-folder", help="the folder in which the training procedure saves the logs", default=LOG_FOLDER)

        self.parser.add_argument("--domains", help="domain folder names (w/o number, but in order)",
                                 default=DOMAINS, nargs="+")

        # --- colla: specific options
        self.parser.add_argument("--input-dropout", help="applies dropout to the input as in the "
                                                         "CollaGAN paper. Can be one from {none (default), "
                                                         "original, aggressive, balanced, conservative, curriculum}",
                                 default="none")
        self.parser.add_argument("--cycled-source-replacer",
                                 help="one from {forward, dropout} indicating which images should be replaced by the "
                                      "forward generated one when computing the cycled images. Colla's paper does not "
                                      "specify this, but its code shows that it replaces all that have been "
                                      "dropped out", default="dropout")
        self.parser.add_argument("--annealing", help="one from {none, linear}, used"
                                                              " to control how the temperature decreases when using"
                                                              " palette quantization", default="none")
        self.parser.add_argument("--temperature", type=float, help="initial temperature for"
                                                                             " the annealing strategy", default=0.1)

        # --- munit and remic: specific options
        self.parser.add_argument("--discriminator-scales", help="Number of scales for the multi-scale "
                                                                "discriminator used in MUNIT and ReMIC", default=3,
                                 type=int)

        # --- all: regarding the datasets to use for training and evaluation
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
        self.parser.add_argument("--tiny-validation", action="store_true", default=False,
                                 help="Uses only tiny (136 test examples) for validation and to generate images in "
                                      "the end")

        self.initialized = True

    def parse(self, args=None, return_parser=False):
        if args is None:
            args = sys.argv[1:]
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args(args)

        if self.values.model != "stargan-unpaired":
            setattr(self.values, "d_steps", 1)
        if self.values.model != "collagan":
            setattr(self.values, "pretrain_epochs", 0)
        if self.values.lambda_l1_backward is None:
            self.values.lambda_l1_backward = self.values.lambda_l1 / 10.
        setattr(self.values, "number_of_domains", len(self.values.domains))
        setattr(self.values, "seed", SEED)
        if self.values.no_aug:
            setattr(self.values, "no_hue", True)
            setattr(self.values, "no_tran", True)
        datasets_used = list(filter(lambda opt: getattr(self.values, opt), ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        setattr(self.values, "datasets_used", datasets_used)
        if len(datasets_used) == 0:
            raise Exception("No dataset was supplied with: --tiny, --rm2k, --rmxp, --rmvx, --misc")
        setattr(self.values, "dataset_names", DATASET_NAMES)
        setattr(self.values, "data_folders", [
            os.sep.join(["datasets", folder])
            for folder
            in self.values.dataset_names
        ])
        setattr(self.values, "domain_folders", [f"{i}-{name}" for i, name in enumerate(self.values.domains)])
        setattr(self.values, "run_string", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        dataset_mask = list(
            map(lambda opt: 1 if getattr(self.values, opt) else 0, ["tiny", "rm2k", "rmxp", "rmvx", "misc"]))
        dataset_sizes = [912, 216, 294, 408, 12372]
        dataset_sizes = [n * m for n, m in zip(dataset_sizes, dataset_mask)]
        train_sizes = [ceil(n * TRAIN_PERCENTAGE) for n in dataset_sizes]
        train_size = sum(train_sizes)
        test_sizes = [dataset_sizes[i] - train_sizes[i]
                      for i, n in enumerate(dataset_sizes)]
        if self.values.rmxp_validation:
            test_sizes = [0, 0, 44, 0, 0]
        elif self.values.rm2k_validation:
            test_sizes = [0, 32, 0, 0, 0]
        elif self.values.rmvx_validation:
            test_sizes = [0, 0, 0, 61, 0]
        elif self.values.tiny_validation:
            test_sizes = [136, 0, 0, 0, 0]

        test_size = sum(test_sizes)

        setattr(self.values, "dataset_sizes", dataset_sizes)
        setattr(self.values, "dataset_mask", dataset_mask)
        setattr(self.values, "train_sizes", train_sizes)
        setattr(self.values, "train_size", train_size)
        setattr(self.values, "test_sizes", test_sizes)
        setattr(self.values, "test_size", test_size)

        setattr(self.values, "pretrain_steps",
                self.values.pretrain_epochs * self.values.train_size // self.values.batch)
        if self.values.steps is None:
            self.values.steps = ceil(self.values.epochs * self.values.train_size / self.values.batch)
        else:
            self.values.epochs = ceil(self.values.steps * self.values.batch / self.values.train_size)

        setattr(self.values, "inner_channels", min(self.values.input_channels, self.values.output_channels))
        setattr(self.values, "domains_capitalized", list(map(lambda name: name[0].upper() + name[1:],
                                                             self.values.domains)))

        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path, argv):
        from io_utils import ensure_folder_structure
        ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(" ".join(argv) + "\n\n")
            file.write(self.get_description("\n", ": ") + "\n")
