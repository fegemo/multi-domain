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
                "callback-evaluate-fid", "callback-evaluate-l1"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 20000,
            "evaluate-steps": 1000,
            "model-name": "@model",
            "experiment": "@dataset,&d-steps,&lr",
        }, {
            "d-steps": [1],
            "lr": [0.0005, 0.0003, 0.0002, 0.00005]
        }, {
            "tiny": {
                "adhoc": ["no-aug"],
            },
            "rm2k": {
                "adhoc": ["no-tran"]
            },
            "rmxp": {
                "adhoc": []
            },
            "rmvx": {
                "adhoc": ["no-tran"]
            },
            "all": {
                "adhoc": ["no-tran"],
                "steps": 80000
            }
        })
    runner.execute(config)
