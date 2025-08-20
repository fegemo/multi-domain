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
                "save-model",
                "palette-quantization",
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 240000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "batch": 16,
            "lr-decay": "constant-then-linear",
            "input-dropout": "original",
            "discriminator-scales": 3,
            "temperature": 4.0,
            "annealing": "cosine",
            "model-name": "@model",
            "experiment": "palette,&lambda-l1,&lambda-latent-reconstruction,&lambda-cyclic-reconstruction,&lambda-palette",
            "vram": -1,
        }, {
            "lambda-l1": [10],
            "lambda-latent-reconstruction": [1, 10],
            "lambda-cyclic-reconstruction": [1, 10, 100],
            "lambda-palette": [0, 1, 10]
        }, {
            "all": {
                "adhoc": ["no-aug"],
            }
        })

    runner.execute(config)
