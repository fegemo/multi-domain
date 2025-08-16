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
            "steps": 6000,
            "evaluate-steps": 100,
            "lr": 0.0001,
            "batch": 4,
            "lr-decay": "constant-then-linear",
            "lambda-l1": 10,
            "lambda-latent-reconstruction": 1,
            "lambda-cyclic-reconstruction": 100,
            "input-dropout": "original",
            "annealing": "linear",
            "model-name": "@model",
            "experiment": "palette,&temperature",
            "vram": -1,
        }, {
            "temperature": [0.01, 0.1, 1., 2., 10.],
        }, {
            "all": {
                "adhoc": ["no-tran"],
            }
        })

    runner.execute(config)
