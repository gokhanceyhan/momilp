"""Implements the executor of the instance generator"""

import argparse
import json
import logging
from src.common.elements import SolverPackage
from src.instance_generator.factory import InstanceCreator, InstanceType

logging.getLogger().setLevel(logging.INFO)


class InstanceGeneratorApp:

    """Implements the command line application for the momilp instance generation"""

    _DATA_FILE_DIR_CONFIGURATION_NAME = "data_file_dir"
    _INSTANCE_TYPE_CONFIGURATION_NAME = "instance_type"
    _NUM_INSTANCES_CONFIGURATION_NAME = "num_instances"
    _PARAMETERS_CONFIGURATION_NAME = "parameters"

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="momilp instance generator app")
        parser.add_argument("-c", "--configuration", help="sets the path to the configuration json file")
        parser.add_argument(
            "-s", "--solver-package", choices=[SolverPackage.GUROBI.value], help="sets the solver package to use")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application
        
        NOTE: Instance generation supports only Gurobi solver currently."""
        args = self._parse_args()
        output_dir = args.working_dir
        with open(args.configuration, mode="r") as f:
            conf = json.load(f)
        instance_type = conf.get(InstanceGeneratorApp._INSTANCE_TYPE_CONFIGURATION_NAME)
        assert instance_type, "the '%s' value must be specified in the configuration file" \
            % InstanceGeneratorApp._INSTANCE_TYPE_CONFIGURATION_NAME
        params = conf.get(InstanceGeneratorApp._PARAMETERS_CONFIGURATION_NAME, {})
        data_file_dir = conf.get(InstanceGeneratorApp._DATA_FILE_DIR_CONFIGURATION_NAME)
        num_instances = conf.get(InstanceGeneratorApp._NUM_INSTANCES_CONFIGURATION_NAME)
        InstanceCreator.create(
            instance_type, output_dir, data_file_dir=data_file_dir, num_instances=num_instances, **params)
