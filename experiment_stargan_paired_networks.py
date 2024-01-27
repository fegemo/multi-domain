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
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 10000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            "lr": 0.0002,
            "lambda-l1": 20.,
            "lambda-palette": 0.,
            "sampler": "multi-target",
            "model-name": "study-networks",
            "experiment": "@dataset,&network,&lambda-l1",
        }, {
            "adhoc": [
                "",
                "source-domain-aware-generator",
                "conditional-discriminator",
                ["source-domain-aware-generator", "conditional-discriminator"]
            ]
        }, {
            # "tiny": {
            #     "adhoc": ["no-aug"]
            # },
            "rm2k": {
                "adhoc": ["no-aug"]
            },
            "rmxp": {
                "adhoc": ["no-aug"]
            },
            # "rmvx": {
            #     "adhoc": ["no-tran"]
            # },
            # "all": {
            #     "adhoc": ["no-tran"],
            #     "steps": 80000
            # }
        })
    runner.execute(config)
