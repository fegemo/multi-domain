#!/usr/bin/env python
import sys

from experiment_runner import Experimenter, create_general_parser

if __name__ == "__main__":
    config = create_general_parser(sys.argv[1:])

    runner = Experimenter(
        "train" if not config.dummy else "dummy_script",
        config.python,
        {
            "model": "sprite",
            "adhoc": [
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "save-model",
                "palette-quantization"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 100000,
            "evaluate-steps": 1000,
            # "capacity": 4,
            "lr": 0.0001,
            # "ttur": 0.2,
            "lr-decay": "constant-then-linear",
            "batch": 4,
            "input-dropout": "conservative",
            "inpaint-mask": "random",
            "generator-scales": 3,
            "discriminator-scales": 3,
            "model-name": "sprite",
            "experiment": "pq,inpd-conservative,inpm-random,gs-3,ds-3,lr-0.0001,ttur-1,&capacity",
            "annealing": "linear",
            "temperature": 0.1,
            "lambda-adversarial": 5.,
            "lambda-reconstruction": 100.,
            "lambda-latent-reconstruction": 1,
            "lambda-kl": 0.01,
            "lambda-palette": 0.1,
            "vram": -1,
        }, {
            "capacity": [4, 3, 2],
        }, {
            "all": {
                "adhoc": ["no-tran"],
            }
        })

    runner.execute(config)
