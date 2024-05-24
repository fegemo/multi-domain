#!/usr/bin/env python3
import logging
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
            "steps": 160000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.0001,
            "batch": 4,
            "lambda-l1": 100,
            "lambda-ssim": 10,
            "lambda-domain": 10,
            "lr-decay": "constant-than-linear",
            "model-name": "@model",
            "experiment": "all,&input-dropout,&cycled-source-replacer"
        }, {
            # "input-dropout": ["none", "original", "aggressive", "balanced", "conservative", "curriculum"],
            "input-dropout": ["conservative"],
            "cycled-source-replacer": ["forward"]
        })

    logging.info("Starting execution of the dropout and replacer experiment.")
    runner.execute(config)

