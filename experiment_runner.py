import hashlib
import json
import os
import re
import subprocess
from functools import reduce
from glob import glob
from graphlib import TopologicalSorter
from itertools import product
from typing import Dict, Any, Union

from tqdm import tqdm

import io_utils


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class Experimenter:
    def __init__(self, script_to_run, path_to_python, default_params, search_grid):
        self.path_to_python = path_to_python
        self.script_to_run = script_to_run
        self.output_path = default_params["log-folder"]
        self.default_params = default_params
        self.search_grid = search_grid

    def run(self):
        checkpoint_file = self.open_checkpoint_file()
        completed, total = self.lookup_start_and_total_runs(checkpoint_file)

        # check if there's a checkpoint and resume from it
        # or start from the beginning
        if completed > 0:
            print(f"Resuming from previous checkpoint {self.hash_checkpoint_name()} that had completed {completed}"
                  f" of {total} runs.")
        else:
            print(f"Starting the experiments from the beginning with hash {self.hash_checkpoint_name()}.")

        combinations = self.explode_combinations()

        try:
            for run_index, combination in tqdm(enumerate(combinations), total=total):
                # skip the first "completed" runs
                if completed > run_index:
                    continue
                elif completed == run_index and run_index != 0:
                    print(f"Skipped the first {completed} runs.")

                # executes the current run
                self.execute_run(combination)

                # saves at the checkpoint file that this test was completed
                self.save_checkpoint(checkpoint_file, run_index + 1, total)

            print(f"All {total} runs are complete.")
            print(f"Head for {self.output_path} to see the logs... hash {self.hash_checkpoint_name()}")
        except ChildProcessError as error:
            print(f"Some error occurred during the execution of runs... exiting!")
            print(f"ChildProcessError at index {run_index} with args {combination}:", error)
        finally:
            # closes the checkpoint file
            checkpoint_file.close()

    def execute_run(self, specific_params):
        run_params = {**self.default_params, **specific_params}
        run_params_string = self.generate_run_params_string(run_params)

        log_file = self.open_log_file(specific_params)
        log_file.seek(0)
        try:
            command = f"{self.path_to_python} {self.script_to_run}.py {run_params_string}"
            log_file.write(command + "\n" * 2)
            process = subprocess.Popen(command, stdout=log_file, stderr=log_file, encoding="utf-8", shell=True)
            process.wait()
            if process.returncode != 0:
                raise ChildProcessError()
        except ChildProcessError as error:
            print(f"Error running command: {command}:", error)
        finally:
            log_file.close()

    def generate_run_params_string(self, run_params):
        interpolated_params = self.interpolate_param_values(run_params)
        run_params_string = ""
        for param_name, param_value in interpolated_params.items():
            positional = False
            adhoc = False
            if param_name == "model":
                positional = True
            elif param_name == "adhoc":
                adhoc = True

            if positional:
                run_params_string += f"{param_value} "
            elif adhoc:
                run_params_string += " ".join(map(lambda v: f"--{v} ", param_value))
            else:
                run_params_string += f"--{param_name} {param_value} "

        return run_params_string

    def interpolate_param_values(self, run_params):
        def get_dependencies(value):
            # check for @....END or @....@ inside value
            if isinstance(value, str):
                tokens = re.split(r"[@&]", value)
                if value[0] != "@" and value[0] != "&":
                    tokens.pop(0)
                return list(filter(lambda t: len(t) > 0, tokens))
            else:
                return []

        interpolated_params = run_params.copy()
        adhoc_params = interpolated_params.pop("adhoc")

        # assemble another dictionary that contains param_names as keys and dependent param_names in a set as values
        dependencies = {name: set(get_dependencies(value)) for name, value in interpolated_params.items()}

        # determines the order in which replacements should occur (through topological sorting)
        ordered_dependencies = tuple(TopologicalSorter(dependencies).static_order())

        # for each value (from least to most dependent), apply the replacement logic
        for param_name in ordered_dependencies:
            for dependency in dependencies[param_name]:
                new_value = interpolated_params[param_name]
                new_value = re.sub(f"@{dependency}",
                                   str(interpolated_params[dependency]),
                                   new_value)
                new_value = re.sub(f"&{dependency}",
                                   dependency.replace('-', '') + str(interpolated_params[dependency]) + ",",
                                   new_value)
                interpolated_params[param_name] = new_value

        # reorders the params to their original order and adds back the adhoc params
        interpolated_params = dict(
            sorted(interpolated_params.items(), key=lambda param: list(run_params.keys()).index(param[0])))
        interpolated_params["adhoc"] = adhoc_params

        return interpolated_params

    def hash_checkpoint_name(self):
        merged_data = {"file": self.script_to_run, **self.default_params, **self.search_grid}
        hashed_data = dict_hash(merged_data)
        return hashed_data[:7]

    def open_checkpoint_file(self):
        checkpoint_path = os.sep.join([self.output_path, self.hash_checkpoint_name()]) + "-ckpt.txt"

        if os.path.isfile(checkpoint_path):
            # the file exists, return a handle
            file = open(checkpoint_path, "r+", encoding="utf-8")
        else:
            # the file DOES not exist yet, create and populate a new one
            io_utils.ensure_folder_structure(self.output_path)
            file = open(checkpoint_path, "w+", encoding="utf-8")
            # the file should have the format like: 25 of 121
            file.write(f"0 of {self.calculate_number_of_runs()}")
            file.flush()
            file.seek(0)
        return file

    def open_log_file(self, specific_params):
        specific_params_string = "-".join(
            [f"{param.replace('-', '')}{value}" for param, value in specific_params.items()]
        )
        log_path = os.sep.join([self.output_path, f"{self.hash_checkpoint_name()}-{specific_params_string}-log.txt"])
        return open(log_path, "w", encoding="utf-8")

    def lookup_start_and_total_runs(self, checkpoint_file):
        # the file should have the format like: 25 of 121
        tokens = next(checkpoint_file).split()
        completed, amount = int(tokens[0]), int(tokens[2])
        checkpoint_file.seek(0)
        return completed, amount

    def save_checkpoint(self, file, completed, total):
        file.write(f"{completed} of {total}")
        file.seek(0)
        file.flush()

    def delete_checkpoint(self):
        files = glob(os.sep.join([self.output_path, f"{self.hash_checkpoint_name()}*.txt"]))
        for path_to_remove in files:
            os.remove(path_to_remove)

        print(f"Deleted {len(files)} files generated on a previous run of this experiment.")

    def calculate_number_of_runs(self):
        combinations = reduce(lambda total, param_list: total * len(param_list), self.search_grid.values(), 1)
        return combinations

    def explode_combinations(self):
        combined_values = list(product(*self.search_grid.values()))
        param_names = list(self.search_grid.keys())
        combinations = [{name: values[i] for i, name in enumerate(param_names)} for values in combined_values]
        return combinations

    def execute(self, config):
        if config.delete:
            self.delete_checkpoint()
            exit(0)
        else:
            self.run()

# example experiments with stargan: d-steps
# runner = Experimenter("train", {
#     "model": "stargan-paired",
#     "adhoc": ["rm2k", "no-aug"],
#     "log-folder": "temp-xper",
#     "epochs": 1,
#     "model-name": "playground",
#     "experiment": "&d-steps&lr"
# }, {
#     "d-steps": [5, 1],
#     "lr": [0.0001, 0.00002]
# })
