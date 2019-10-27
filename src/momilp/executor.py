"""Implements the executor class to run the momilp solver"""

import argparse


class Executor:

    """Implements the momilp solver executor"""

    def __init__(self, model):
        self._model = model

    def execute(self):
        """Executes the momilp solver"""
        pass


class MomilpSolverApp:

    """Implements the command line application for the momilp solver executor"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="momilp solver app")
        parser.add_argument("-m", "--model-file-path", help="sets the path to the model file")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        print(args.model_file_path)
        print(args.working_dir)
