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
                "conditional-discriminator", "source-domain-aware-generator",
                "save-model",
                "palette-quantization"
            ],
            "log-folder": config.output if config.output is not None else "output",
            "steps": 80000,
            "evaluate-steps": 1000,
            "d-steps": 1,
            "lr": 0.0002,
            "batch": 4,
            "capacity": 4,
            "conditional_discriminator": True,
            "lambda_domain": 1.0,
            "lambda_gp": 10.0,
            "lambda_l1": 100.0,
            "lambda_l1_backward": 10.0,
            "lambda_palette": 0.0,
            "lr_decay": "constant-then-linear",
            "no_aug": False,
            "no_hue": False,
            "no_tran": True,
            "sampler": True,
            "source_domain_aware_generator": True,
            "rm2k": True,
            "rmxp": True,
            "rmvx": True,
            "tiny": True,
            "misc": True,
            "annealing": "linear",
            "temperature": 0.1,
            "lambda-palette": 1.0,
            "model-name": "@model",
            "experiment": "@dataset, &adhoc, &resizing-factor",
            "vram": -1
        },
        {
            "adhoc": [ #?
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