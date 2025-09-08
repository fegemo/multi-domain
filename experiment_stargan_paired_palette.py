#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "stargan-paired",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1", #"callback-debug-discriminator",
                "conditional-discriminator", "source-domain-aware-generator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            #"steps": 40000,
            "steps": 100000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            #"lr": 0.0001,
            "lr": 0.0002,
            "batch": 4, #new
            "capacity": 4, #new
            #"lambda-l1": 400.,
            "lambda-l1": 100.0,
            "model-name": "@model",
            "experiment": "palette,@dataset",
            "lambda-domain": 1.0, #new
            "lambda-gp": 10.0, #new
            "lambda-l1-backward": 10.0, #new
            "lambda-palette": 1.0, #new
            "lr-decay": "constant-then-linear", #new
            "vram": -1 #new
        },
        {},
        {
            "all": {
                "adhoc": ["no-tran"]
            }
            # "tiny": {
            #     "adhoc": ["no-aug"]
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
            # "all": {
            #     "adhoc": ["no-tran"],
            #     "steps": 80000
            # }
        })
    runner.execute(config)
