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
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 240000,
            "evaluate-steps": 1000,
            "lr": 0.0001,
            "lr-decay": "constant-then-linear",
            "batch": 4,
            "input-dropout": "conservative",
            "capacity": 4,
            "cycled-source-replacer": "forward",
            "lambda-l1": 100.,
            "lambda-ssim": 10.,
            "lambda-domain": 10.,
            "model-name": "@model",
            "experiment": "@dataset,&annealing,&temperature,&lambda-palette",
            "annealing": "linear",
            "generator": "palette",
        }, {
            # "temperature": [1, 0.1, 0.01],
            "lambda-palette": [1., 10., 100.],
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
            }
        })

    runner.execute(config)
