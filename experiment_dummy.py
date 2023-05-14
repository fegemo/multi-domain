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
        "dummy_script",
        config.python,
        {
            "model": "stargan-paired",
            "adhoc": ["rm2k", "no-aug", "callback-evaluate-fid", "callback-evaluate-l1"],
            "log-folder": config.output if config.output is not None else "output/dsteps-study",
            "epochs": 240,
            "model-name": "@model",
            "experiment": "&d-steps&lr",
        }, {
            "d-steps": [1],
            "lr": [0.0001],
            "l1": [10., 1., 20., 30., 50., 100.]
        })
    runner.execute(config)
