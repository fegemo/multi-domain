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
            "lambda-l1": 10,
            "lambda-latent-reconstruction": 1,
            "lambda-palette": 1,
            "input-dropout": "original",
            "model-name": "@model",
            "experiment": "palette,&annealing,&temperature,&lambda-cyclic-reconstruction",
            "vram": -1,
        }, {
            "temperature": [1., 4., 10.],
            "lambda-cyclic-reconstruction": [1, 100],
            "annealing": ["linear", "cosine"],
        }, {
            "all": {
                "adhoc": ["no-tran"],
            }
        })

    runner.execute(config)
