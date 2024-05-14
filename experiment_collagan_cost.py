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
            "steps": 40000,
            "evaluate-steps": 1000,
            "capacity": 2,
            "d-steps": 1,
            "lr": 0.0001,
            "batch": 4,
            # "lambda-l1-backward": 10,  # if omitted, automatically calculated as lambda_l1/10.
            "lr-decay": "constant-than-linear",
            "input-dropout": "original",
            "cycled-source-replacer": "forward",
            "model-name": "@model",
            "experiment": "@dataset,&lambda-l1,&lambda-ssim,&lambda-domain"
        }, {
            "lambda-l1": [20, 100],
            "lambda-ssim": [0, 1, 10, 100],
            "lambda-domain": [0, 1, 10, 100]
        }, {
            # "tiny": {
            #     "adhoc": ["no-aug"],
            # },
            "rm2k": {
                "adhoc": ["no-tran"]
            },
            "rmxp": {
                "adhoc": []
            },
            # "rmvx": {
            #     "adhoc": ["no-tran"]
            # },
            # "all": {
            #     "adhoc": ["no-tran"],
            # }
        })

    runner.execute(config)
