#!/usr/bin/env python3
import argparse
import sys

from experiment_runner import Experimenter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", "-d", help="Instead of training, deletes the checkpoint and log files"
                                               "for this experiment", action="store_true")
    parser.add_argument("--output", "-o", help="Sets (overrides) the path to the output folder", default=None)
    parser.add_argument("--python", "-p", help="Path to python with tensorflow", default="venv/Scripts/python")
    config = parser.parse_args(sys.argv[1:])

    runner = Experimenter(
        "train",
        config.python,
        {
            "model": "stargan-paired",
            "adhoc": [
                "rm2k",
                "no-tran",
                "callback-evaluate-fid", "callback-evaluate-l1", "callback-debug-discriminator",
                "conditional-discriminator", "source-domain-aware-generator",
                "save-model"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "epochs": 500,
            "d-steps": 1,
            "lr": 0.0001,
            "lambda-l1": 100.,
            "lambda-palette": 100.,
            "sampler": "multi-target",
            "model-name": "@model",
            "experiment": "rm2k@model&lambda-tv",
        }, {
            "lambda-tv": [0, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
        })
    runner.execute(config)