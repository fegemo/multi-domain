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
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 480000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.00001,
            "ttur": 0.5,
            "lr-decay": "constant-then-linear",
            "batch": 20,
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            "model-name": "collagan-palette-coverage",
            "experiment": "lr-1e-5,batch-20,&lambda-palette,ttur-0.5",
            "generator": "palette",
            "temperature": 0.1,
            "annealing": "linear",
            "lambda-l1": 100.,
            "lambda-ssim": 1.,
            "lambda-domain": 10.,
            "lambda-regularization": 0.001,
            "lambda-histogram": 0.,
            "vram": -1,
        }, {
            "lambda-palette": [1.],
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
