#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "remic",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 90000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "batch": 16,
            "lr-decay": "none",
            "input-dropout": "original",
            "discriminator-scales": 1,
            "model-name": "@model",
            "experiment": "@dataset,&lambda-l1,&lambda-latent-reconstruction,&lambda-cyclic-reconstruction",
        }, {
            "lambda-l1": [10, 100],
            "lambda-latent-reconstruction": [0, 1, 10],
            "lambda-cyclic-reconstruction": [0, 20, 200],
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
                # "steps": 80000
            }
        })

    runner.execute(config)