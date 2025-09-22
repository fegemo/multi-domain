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
                "callback-evaluate-fid", "callback-evaluate-l1",
                "save-model",
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 240000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "batch": 4,
            "lr-decay": "constant-then-linear",
            "input-dropout": "original",
            "discriminator-scales": 3,
            "model-name": "@model",
            "experiment": "&lambda-l1,&lambda-latent-reconstruction,&lambda-cyclic-reconstruction",
            "vram": -1,
            "patience": 20,
        }, {
            "lambda-l1": [10],
            "lambda-latent-reconstruction": [1],
            "lambda-cyclic-reconstruction": [0, 1, 10, 100]
        }, {
            "all": {
                "adhoc": ["no-aug"],
            }
        })

    runner.execute(config)
