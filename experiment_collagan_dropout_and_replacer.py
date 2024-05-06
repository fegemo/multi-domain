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
                "callback-evaluate-fid", "callback-evaluate-l1",
                "save-model",
                "all", "no-tran"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 40000,
            "evaluate-steps": 1000,
            "capacity": 2,
            "lr": 0.0001,
            "batch": 4,
            "lambda-l1": 20,
            "lambda-ssim": 100,
            "lambda-domain": 0.5,
            "lr-decay": "constant-than-linear",
            "model-name": "@model",
            "experiment": "all,&dropout,&cycled-source-replacer"
        }, {
            "adhoc": ["aggressive-input-dropout", "input-dropout", ""],
            "cycled-source-replacer": ["dropout", "forward"]
        })

    runner.execute(config)
