import argparse
import hashlib
import json
import os
import re
import subprocess
from functools import reduce
from glob import glob
from graphlib import TopologicalSorter
from itertools import product
from typing import List
from copy import deepcopy

from tqdm import tqdm
from beaupy import select_multiple

from utility import io_utils
from utility.functional_utils import listify


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_human_readable_combination(combination_dict):
    return ", ".join([f"{k}-{v}" for k, v in combination_dict.items()])


class Experimenter:
    valid_datasets = ["tiny", "rm2k", "rmxp", "rmvx", "misc"]
    virtual_datasets = valid_datasets + ["all"]
    artificial_param_lookup_table = {
        "all": valid_datasets
    }
    valid_networks = ["source-domain-aware-generator", "conditional-discriminator"]

    def __init__(self, script_to_run, path_to_python, default_params, search_grid, dataset_params=None):
        if "adhoc" not in default_params:
            default_params["adhoc"] = []

        if "adhoc" not in search_grid:
            search_grid["adhoc"] = []

        if dataset_params is None or dataset_params == {}:
            # if no dataset params are specified, then we look for dataset names used in the adhoc params
            dataset_specified_in_default_params = [n for n in default_params["adhoc"] if n in self.virtual_datasets]
            dataset_specified_in_specific_params = [n for n in search_grid["adhoc"] if n in self.virtual_datasets]
            specified_datasets = [*dataset_specified_in_default_params, *dataset_specified_in_specific_params]
            dataset_params = {n: {} for n in specified_datasets}

        # we remove the dataset names from the adhoc params, as we suppose they were provided (or either transported)
        # to the dataset_params
        default_params["adhoc"] = [n for n in default_params["adhoc"] if n not in self.virtual_datasets]
        search_grid["adhoc"] = [n for n in search_grid["adhoc"] if n not in self.virtual_datasets]

        self.path_to_python = path_to_python
        self.script_to_run = script_to_run
        self.output_path = default_params["log-folder"]
        self.default_params = default_params
        self.search_grid = search_grid
        self.dataset_params = dataset_params
        self.datasets = list(dataset_params.keys())

    def run(self, selected_combinations=None):
        all_combinations = self.explode_combinations()
        selected_combinations = selected_combinations or all_combinations

        execution_status, total = self.lookup_start_and_total_runs()
        completed_runs = reduce(lambda count, status: count + 1 if status else count, execution_status, 0)

        # check if there's a checkpoint and resume from it
        # or start from the beginning
        if completed_runs > 0:
            print(f"Resuming from previous checkpoint {self.hash_checkpoint_name()} that had completed {completed_runs}"
                  f" of {total} runs: {execution_status}.")
        elif completed_runs == total:
            print("All runs WERE ALREADY completed.")
            return
        else:
            print(f"Starting the experiments from the beginning with hash {self.hash_checkpoint_name()}.")

        run_index = -1
        combination = "''"
        try:
            pbar = tqdm(enumerate(all_combinations), total=total, unit="test", dynamic_ncols=True)
            for run_index, combination in pbar:
                pbar.set_description(f"Params: {get_human_readable_combination(combination)}")
                execution_status, _ = self.lookup_start_and_total_runs()
                if execution_status[run_index]:
                    print("Skipped the run with index", run_index, "because it was already completed.")
                    continue
                if combination not in selected_combinations:
                    print("Skipped the run with index", run_index, "because it was not selected.")
                    continue

                # executes the current run
                success = self.execute_run(combination)
                execution_status[run_index] = success

                # saves at the checkpoint file that this test was completed
                self.save_checkpoint(execution_status, total)

            all_success = bool(reduce(lambda a, b: a and b, execution_status, True))
            all_failure = bool(reduce(lambda a, b: not a and not b, execution_status, True))
            execution_status_description = "successfully" if all_success else \
                ("all with errors" if all_failure else "with some errors")
            print(f"All {total} runs are complete with: {execution_status_description}.")
            print(f"Head for {self.output_path} to see the logs... hash {self.hash_checkpoint_name()}")
        except ChildProcessError as error:
            print(f"ChildProcessError at index {run_index} with args {combination}:", error)
            print(f"Some error occurred during the execution of runs... exiting!")

    def combine_default_and_specific_params(self, default_params, specific_params):
        """
        Combines the default params with the specific ones, doing a union of the values in case of lists.
        :param specific_params:
        :return:
        """
        combined_params = deepcopy(default_params)
        for param_name, param_value in specific_params.items():
            if param_name in specific_params:
                if param_name not in default_params or not isinstance(default_params[param_name], list):
                    combined_params[param_name] = param_value
                else:
                    combined_params[param_name] += param_value
        return combined_params

    def combine_params_with_dataset_params(self, run_params):
        current_dataset_name = [name for name in self.datasets if name in run_params["adhoc"]][0]
        current_dataset_params = self.dataset_params[current_dataset_name]

        return self.combine_default_and_specific_params(run_params, current_dataset_params)

    def replace_artificial_params(self, run_params):
        new_run_params = {**run_params}
        for param_name, param_value in run_params.items():
            if isinstance(param_value, list):
                new_param_value = []
                for i, value in enumerate(param_value):
                    if value in self.artificial_param_lookup_table:
                        new_param_value += listify(self.lookup_artificial_param(value))
                    else:
                        new_param_value.append(value)
                new_run_params[param_name] = new_param_value
            else:
                new_run_params[param_name] = self.lookup_artificial_param(param_value)
        return new_run_params

    def execute_run(self, specific_params):
        run_params = self.combine_default_and_specific_params(self.default_params, specific_params)
        run_params = self.combine_params_with_dataset_params(run_params)
        run_params = self.replace_artificial_params(run_params)
        run_params_string = self.generate_run_params_string(run_params, specific_params)

        log_file = self.open_log_file(specific_params)
        log_file.seek(0)
        command = "''"
        try:
            command = f"{self.path_to_python} {self.script_to_run}.py {run_params_string}"
            log_file.write(command + "\n" * 2)
            process = subprocess.Popen(command, stdout=log_file, stderr=log_file, encoding="utf-8", shell=True)
            process.wait()
            if process.returncode != 0:
                raise ChildProcessError()
            return True
        except ChildProcessError as error:
            print(f"Error running command: {command}:", error)
            return False
        finally:
            log_file.close()

    def lookup_artificial_param(self, name):
        if name in self.artificial_param_lookup_table:
            return self.artificial_param_lookup_table[name]
        else:
            return name

    def generate_run_params_string(self, run_params, specific_params):
        interpolated_params = self.interpolate_param_values(run_params, specific_params)
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
                run_params_string += " ".join([f"--{v}" for v in param_value if v != ""]) + " "
            else:
                run_params_string += f"--{param_name} {param_value} "

        return run_params_string

    def interpolate_param_values(self, run_params, specific_params):
        def get_dependencies(value):
            # dependencies are specified as @param_name (value only) or &param_name (name + '-' + value)
            # check for @....END or @....@ inside value
            if isinstance(value, str):
                tokens = re.split(r"[@&]", value)
                # remove the optional , at the end of each token
                for i, token in enumerate(tokens):
                    while len(tokens[i]) > 0 and tokens[i][-1] == ",":
                        tokens[i] = tokens[i][:-1]
                # if the first token is not a dependency, simply ignore it
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
            if param_name not in dependencies:
                # it is okay -- param_name might be a virtual parameter such as database or adhoc
                continue
            for dependency in dependencies[param_name]:
                if dependency in interpolated_params:
                    replaced_value = interpolated_params[dependency]
                elif dependency == "dataset":
                    dataset = [ds for ds in self.datasets if ds in specific_params["adhoc"]][0]
                    replaced_value = dataset
                elif dependency == "network":
                    network_config = [n for n in self.valid_networks if n in specific_params["adhoc"]]
                    if len(network_config) == 2:
                        network_config = "both"
                    elif len(network_config) == 0:
                        network_config = "none"
                    else:
                        network_config = network_config[0]
                    replaced_value = network_config
                elif dependency == "adhoc":
                    adhoc_without_dataset = [n for n in specific_params["adhoc"] if n not in self.datasets]
                    replaced_value = ",".join(adhoc_without_dataset)
                else:
                    replaced_value = "???"

                if isinstance(replaced_value, list):
                    replaced_value = ",".join(replaced_value)

                new_value = interpolated_params[param_name]
                new_value = re.sub(f"@{dependency}",
                                   str(replaced_value),
                                   new_value)
                new_value = re.sub(f"&{dependency}",
                                   dependency + "-" + str(replaced_value),
                                   new_value)
                if new_value[0] == "-":
                    new_value = new_value[1:]
                if new_value[-1] == ",":
                    new_value = new_value[:-1]
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
            # the file should have the format like: ooox.. (3 success, 1 error, 2 not run)
            file.write("".join(["."] * self.calculate_number_of_runs()))
            file.flush()
            file.seek(0)
        return file

    def open_log_file(self, specific_params):
        def value_stringifier(value):
            if isinstance(value, list):
                non_empty_values = [v for v in value if v != ""]
                return ",".join(non_empty_values)
            else:
                return str(value)

        specific_params_string = "-".join(
            [f"{param.replace('adhoc', '').replace('-', '')}{value_stringifier(value)}" for param, value in
             specific_params.items()]
        )
        log_path = os.sep.join([self.output_path, f"{self.hash_checkpoint_name()}-{specific_params_string}-log.txt"])
        return open(log_path, "w", encoding="utf-8")

    def lookup_start_and_total_runs(self):
        # the file should have the format like: ooox.. (3 successes, 1 error, 2 not run)
        checkpoint_file = self.open_checkpoint_file()
        execution_status = next(checkpoint_file)
        execution_status = list(map(lambda s: True if s == "o" else False, execution_status))
        checkpoint_file.seek(0)
        checkpoint_file.close()
        return execution_status, len(execution_status)

    # def save_checkpoint(self, file, completed, total):
    def save_checkpoint(self, execution_status: List[bool], total):
        file = self.open_checkpoint_file()
        execution_status = list(map(lambda s: "o" if s else "x", execution_status))
        execution_status += ["."] * (total - len(execution_status))
        execution_status = "".join(execution_status)
        file.write(execution_status)
        file.seek(0)
        file.flush()
        file.close()

    def delete_checkpoint(self):
        files = glob(os.sep.join([self.output_path, f"{self.hash_checkpoint_name()}*.txt"]))
        for path_to_remove in files:
            os.remove(path_to_remove)

        print(f"Deleted {len(files)} files generated on a previous run of this experiment.")

    def calculate_number_of_runs(self):
        combinations = reduce(lambda total, param_list: total * max(1, len(param_list)), self.search_grid.values(), 1)
        combinations *= len(self.datasets)
        return combinations

    def fuse_equal_params(self, param_names, param_values):
        """
        Looks for equal names of parameters in param_names, and fuses (add to a list) them and their values
        :param param_names: list of names of parameters
        :param param_values: list of values of parameters
        :return: a tuple containing the fused param_names and the fused param_values
        """
        new_param_names = []
        new_param_values = [list(v) for v in param_values]
        for i, name in enumerate(param_names):
            if name not in new_param_names:
                new_param_names += [name]
            else:
                index = new_param_names.index(name)
                for j, combination in enumerate(new_param_values):
                    new_param_values[j][index] = listify(new_param_values[j][index]) + listify(new_param_values[j][i])
                    new_param_values[j].pop(i)

        return new_param_names, new_param_values

    def explode_combinations(self):
        non_empty_search_params = {name: values for name, values in self.search_grid.items() if len(values) > 0}
        combined_values = list(product(self.datasets, *non_empty_search_params.values()))
        combined_values = [tuple([[value[0]]] + list(value[1:])) for value in combined_values]
        param_names = ["adhoc"] + list(non_empty_search_params.keys())
        param_names, combined_values = self.fuse_equal_params(param_names, combined_values)
        combinations = [{name: values[i] for i, name in enumerate(param_names)} for values in combined_values]
        return combinations

    def show_interactive_menu(self):
        stringify_shallow_list = lambda ls: " ".join([str(x) for x in ls]) if isinstance(ls, list) else str(ls)
        stringify_shallow_dict = lambda d: " ".join([f"{k}-{stringify_shallow_list(v)}" for k, v in d.items()])
        all_combinations = self.explode_combinations()
        print("Choose which configurations to run:")
        selected_combinations = select_multiple(all_combinations, preprocessor=lambda x: stringify_shallow_dict(x))
        return selected_combinations

    def execute(self, config):
        if config.delete:
            self.delete_checkpoint()
            exit(0)

        if config.nofid:
            fid_command = "callback-evaluate-fid"
            if fid_command in self.default_params["adhoc"]:
                self.default_params["adhoc"].remove(fid_command)

        if config.interactive:
            print("Staring interactive mode...")
            selected = self.show_interactive_menu()
            self.run(selected_combinations=selected)
        else:
            self.run()


def create_general_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", "-d", help="Instead of training, deletes the checkpoint and log files"
                                               "for this experiment", action="store_true")
    parser.add_argument("--output", "-o", help="Sets (overrides) the path to the output folder", default=None)
    parser.add_argument("--python", "-p", help="Path to python with tensorflow", default="python")
    parser.add_argument("--dummy", "-D", help="Dummy run, does not execute anything", action="store_true")
    parser.add_argument("--interactive", "-i", help="Interactive mode, asks for which combinations of "
                                                    "the search grid should be run", default=False, action="store_true")
    parser.add_argument("--nofid", "-f", help="Do NOT calculate FID", action="store_true", default=False)
    config = parser.parse_args(args)
    return config

# example experiments with stargan: d-steps
# runner = Experimenter("train", {
#     "model": "stargan-paired",
#     "adhoc": ["conditional-discriminator"],
#     "log-folder": "temp-xper",
#     "epochs": 1,
#     "model-name": "playground",
#     "experiment": "&d-steps&lr"
# }, {
#     "d-steps": [5, 1],
#     "lr": [0.0001, 0.00002]
# }, {
#     "tiny": {
#          "adhoc": ["--no-aug"]
#     },
#     "all": {
#          "adhoc": ["no-tran"]
#     }
# })
