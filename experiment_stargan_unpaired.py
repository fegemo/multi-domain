#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "stargan-unpaired",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 40000,
            "evaluate-steps": 1000,
            "d-steps": 5,
            "lr": 0.0001,
            "lambda-l1": 0.,
            "model-name": "@model",
            "experiment": "@dataset",
            "sampler": "single-target"
        }, {}, {
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
            "all": {
                "adhoc": ["no-tran"]
            }
        })

    runner.execute(config)
