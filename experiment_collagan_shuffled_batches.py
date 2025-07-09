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
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "save-model",
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 300000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.0001,
            "ttur": 0.2,
            "lr-decay": "none",
            "batch": 4,
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            "model-name": "collagan-shuffled-batches",
            "experiment": "&adhoc",
            "lambda-adversarial": 1.,
            "lambda-l1": 100.,
            "lambda-ssim": 10.,
            "lambda-domain": 10.,
            "lambda-regularization": 0.001,
            "lambda-histogram": 0.,
            "lambda-palette": 1.0,
            "vram": -1,
        }, {
            "adhoc": [["shuffled-batches"], [""]],
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
