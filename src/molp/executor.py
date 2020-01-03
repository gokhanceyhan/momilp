"""Implements the executor class to run the molp solver"""

import argparse
from enum import Enum
from gurobipy import Model, read
from src.common.elements import SolverPackage
from src.molp.factory import MolpSolverFactory
 

class Executor:

    """Implements the molp solver executor"""

    _MODEL_CLASS_TO_SOLVER_PACKAGE = {Model: SolverPackage.GUROBI}
    _SUPPORTED_SOLVER_PACKAGES = [SolverPackage.GUROBI]
    _UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE = \
        "the solver package is not supported, define the model in one of the '{supported_solvers!s}' solver packages"

    def __init__(self, model):
        self._model = model
        # NOTE: for some reason, the solver package below does not return as an enum type
        solver_package = Executor._MODEL_CLASS_TO_SOLVER_PACKAGE[model.__class__]
        if solver_package not in Executor._SUPPORTED_SOLVER_PACKAGES:
            error_message = Executor._UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE.format(
                solver=solver_package.value, supported_solvers=[p.value for p in Executor._SUPPORTED_SOLVER_PACKAGES])
            raise ValueError(error_message)
        self._solver_package = solver_package

    def execute(self):
        """Executes the molp solver"""
        model = self._model
        solver = MolpSolverFactory.create_solver(model, 2, solver_package=self._solver_package)
        solver.solve()
        return solver.extreme_supported_nondominated_points()


class MolpSolverApp:

    """Implements the command line application for the molp solver executor"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="molp solver app")
        parser.add_argument("-m", "--model-file-path", help="sets the path to the model file (.lp format)")
        parser.add_argument(
            "-s", "--solver-package", choices=[SolverPackage.GUROBI.value], help="sets the solver package to use")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        model = read(args.model_file_path)
        executor = Executor(model)
        executor.execute()