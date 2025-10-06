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
                "callback-evaluate-fid",
                "callback-evaluate-l1",
                #"callback-debug-discriminator",
                "save-model",
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 100000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.00001,
            "ttur": 1,
            "batch": 1,
            "lr-decay": "constant-then-linear",
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            "model-name": "@model",
            "experiment": "upscaling-augmentation-2,@dataset",
            "generator": "affluent",
            "lambda-adversarial": 1.,
            "lambda-l1": 100.,
            "lambda-ssim": 1.,
            "lambda-domain": 10.,
            "lambda-regularization": 0.001,
            "vram": -1,
            "resizing-factor": 3
        },
        {},
        {
            "all": {
                "adhoc": ["no-tran"]
            }
        }
    )

    runner.execute(config)
