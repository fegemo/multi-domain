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
            "steps": 300000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "d-steps": 1,
            "lr": 0.00011,
            "batch": 4,
            "lr-decay": "constant-than-linear",
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            "model-name": "@model",
            "experiment": "@dataset,annealing,&lambda-l1,&lambda-ssim,&lambda-domain,&lambda-palette",
            "generator": "palette",
            "temperature": 0.1,
            "annealing": "linear",
        }, {
            "lambda-l1": [100],
            "lambda-ssim": [1, 10],
            "lambda-domain": [10],
            "lambda-palette": [0, 100, 1000],
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
