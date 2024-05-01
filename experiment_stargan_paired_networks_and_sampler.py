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
                "callback-evaluate-fid", "callback-evaluate-l1",
                "save-model",
                "all", "no-tran"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 40000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            "lr": 0.0002,
            "lambda-l1": 100.,
            "model-name": "study-networks",
            "experiment": "all,&network,&sampler",
        }, {
            "adhoc": [
                "",
                "source-domain-aware-generator",
                "conditional-discriminator",
                ["source-domain-aware-generator", "conditional-discriminator"]
            ],
            "sampler": ["single-target", "multi-target"]
        })
    runner.execute(config)
