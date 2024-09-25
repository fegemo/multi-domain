#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "munit",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 40000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "batch": 1,
            "lr-decay": "none",
            "model-name": "@model",
            "experiment": "@dataset,l1@lambda-l1,latent@lambda-latent-reconstruction"
        }, {
            "lambda-l1": [1, 10, 100, 400],
            "lambda-latent-reconstruction": [0, 1, 10, 100]
        }, {
            "tiny": {
                "adhoc": ["no-aug"],
            },
            "rm2k": {
                "adhoc": ["no-tran"]
            },
            "rmxp": {
                "adhoc": []
            },
            "rmvx": {
                "adhoc": ["no-tran"]
            },
            "all": {
                "adhoc": ["no-tran"],
                "steps": 80000
            }
        })

    runner.execute(config)
