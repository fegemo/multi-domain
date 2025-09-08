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
                "callback-evaluate-fid", "callback-evaluate-l1", #"callback-debug-discriminator",
                "conditional-discriminator", "source-domain-aware-generator",
                "save-model",
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 100000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            "lr": 0.0002,
            "batch": 4,
            "capacity": 4,
            "lambda-domain": 1.0,
            "lambda-gp": 10.0,
            "lambda-l1": 100.0,
            "lambda-l1-backward": 10.0,
            "lr-decay": "constant-then-linear",
            "model-name": "@model",
            "experiment": "resizing,@dataset,&adhoc,&resizing-factor",
            "vram": -1
        },
        {
            "adhoc": [ #?
                ["up-preprocessing", "no-up-aug"], #só pre-processamento
                [], #só augmentation
                ["up-preprocessing"], #ambos
            ],
            "resizing-factor": [2, 3]
        },
        {
            "all": {
                "adhoc": ["no-tran"]
            }
        }
    )

    runner.execute(config)