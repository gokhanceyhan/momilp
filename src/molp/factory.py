"""Implements multi-objective linear programming solver factory"""

from enum import Enum
from src.common.elements import SolverPackage
from src.molp.dichotomic_search.solver import BolpDichotomicSearchWithGurobiSolver


class SolverType(Enum):

    """Represents solver type"""

    DICHOTOMIC_SEARCH = "dichotomic search"
    PARAMETRIC_SIMPLEX = "parametric simplex"


class MolpSolverFactory:

    """Implements multi-objective linear programming (MOLP) solver factory"""

    _SUPPORTED_NUM_OBJECTIVES = [2]
    _SUPPORTED_SOLVER_PACKAGES = [SolverPackage.GUROBI]
    _SUPPORTED_SOLVER_TYPES = [SolverType.DICHOTOMIC_SEARCH]
    _UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE = \
        "the '%s'-objective problems are not supported, the supported number of objectives are '%s'"
    _UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE = \
        "the '%s' solver package is not supported, the supported solver packages are '%s'"
    _UNSUPPORTED_SOLVER_TYPE_ERROR_MESSAGE = \
        "the '%s' solver type is not supported, the supported solver types are '%s'"

    @staticmethod
    def create_solver(
            model, num_objectives, solver_package=SolverPackage.GUROBI, solver_type=SolverType.DICHOTOMIC_SEARCH):
        """Creates and returns the molp solver"""
        if num_objectives not in MolpSolverFactory._SUPPORTED_NUM_OBJECTIVES:
            raise ValueError(
                MolpSolverFactory._UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE % (
                    num_objectives, MolpSolverFactory._SUPPORTED_NUM_OBJECTIVES))
        if solver_package not in MolpSolverFactory._SUPPORTED_SOLVER_PACKAGES:
            raise ValueError(
                MolpSolverFactory._UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE % (
                    solver_package.value, [package.value for package in MolpSolverFactory._SUPPORTED_SOLVER_PACKAGES]))
        if solver_type not in MolpSolverFactory._SUPPORTED_SOLVER_TYPES:
            raise ValueError(
                MolpSolverFactory._UNSUPPORTED_SOLVER_TYPE_ERROR_MESSAGE % (
                    solver_type.value, [type_.value for type_ in MolpSolverFactory._SUPPORTED_SOLVER_TYPES]))
        return BolpDichotomicSearchWithGurobiSolver(model)


