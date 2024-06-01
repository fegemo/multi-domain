#!/usr/bin/env python3
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "stargan-paired",
            "adhoc": [
                "callback-evaluate-l1",
                "conditional-discriminator", "source-domain-aware-generator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 80000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            "lr": 0.0002,
            "sampler": "multi-target",
            "model-name": "@model",
            "experiment": "@dataset,&lambda-l1,&lambda-domain,&lambda-reconstruction&capacity",
        }, {
            "lambda-l1": [100.],
            "lambda-domain": [10., 1.],
            "lambda-reconstruction": [10., 100.],
            "capacity": [4, 2, 1],
        }, {
            # "tiny": {
            #     "adhoc": ["no-aug"]
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
                # "steps": 80000
            }
        })
    runner.execute(config)
