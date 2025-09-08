#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "collagan",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1", #"callback-debug-discriminator",
                "save-model",
                "palette-quantization" #new
            ],
            "log-folder": config.output if config.output is not None else "output",
            #"steps": 300000,
            "steps": 100000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.0001,
            #"ttur": 0.2,
            "ttur": 1,
            "lr-decay": "constant-then-linear",
            "batch": 4,
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            #"model-name": "collagan-palette-coverage",
            "model-name": "@model",
            #"experiment": "lambda-palette-1.0,&temperature",
            "experiment": "palette,@dataset,&adhoc",
            #"generator": "palette",
            "generator": "affluent",
            "annealing": "linear",
            "lambda-adversarial": 10.,
            #"lambda-l1": 150.,
            "lambda-l1": 100.,
            #"lambda-ssim": 10.,
            "lambda-ssim": 1.,
            "lambda-domain": 10.,
            "lambda-regularization": 0.001,
            #"lambda-histogram": 0.,
            "lambda-palette": 1.0,
            "temperature": 0.1, #new
            "vram": -1,
        }, {
            #"temperature": [1.0, 0.01],
        }, {
            # "tiny": {
            #     "adhoc": ["no-aug"],
            # },
            # "rm2k": {
            #     "adhoc": ["no-tran"]
            # },
            # "rmxp": {
            #     "adhoc": []
            # },
            # "rmvx": {
            #     "adhoc": ["no-tran"]
            # },
            "all": {
                "adhoc": ["no-tran"],
            }
        })

    runner.execute(config)
