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
            "steps": 240000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "batch": 16,
            "lr-decay": "none",
            "lambda-l1": 10,
            "lambda-latent-reconstruction": 1,
            "lambda-cyclic-reconstruction": 100,
            "lambda-gp": 0,
            "discriminator-scales": 3,
            "input-dropout": "original",
            "model-name": "@model",
            "experiment": "baseline",
        }, {
        }, {
            "all": {
                "adhoc": ["no-tran"],
            }
        })

    runner.execute(config)
