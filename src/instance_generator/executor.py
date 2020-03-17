"""Implements the executor of the instance generator"""

import argparse
import json
from src.common.elements import SolverPackage
from src.instance_generator.factory import InstanceCreator, InstanceType


class InstanceGeneratorApp:

    """Implements the command line application for the momilp instance generation"""

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
        instance_type = conf["instance_type"]
        params = conf["parameters"]
        num_instances = conf["num_instances"]
        InstanceCreator.create(instance_type, num_instances, output_dir, **params)
