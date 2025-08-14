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
                "callback-debug-discriminator",
                "save-model",
                "palette-quantization"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 240000,
            "evaluate-steps": 1000,
            "capacity": 4,
            "lr": 0.0001,
            "ttur": 1,
            "batch": 4,
            "annealing": "linear",
            "temperature": 0.1,
            "lr-decay": "constant-then-linear",
            "input-dropout": "conservative",
            "cycled-source-replacer": "forward",
            "model-name": "@model",
            "experiment": "@dataset, &adhoc, &resizing-factor",
            "generator": "affluent",
            "lambda-adversarial": 10.,
            "lambda-l1": 100.,
            "lambda-ssim": 1.,
            "lambda-domain": 10.,
            "lambda-regularization": 0.001,
            "lambda-palette": 1.0,
            "vram": -1
        },
        {
            "adhoc": [
                ["up-preprocessing", "no-up-aug"], #só pre-processamento
                [], #só augmentation
                ["up-preprocessing"], #ambos
            ],
            "resizing-factor": [1, 2, 3]
        },
        {
            "all": {
                "adhoc": ["no-tran"]
            }
        }
    )

    runner.execute(config)