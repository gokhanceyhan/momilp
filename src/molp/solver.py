"""Implements multi-objective linear programming solver"""

import abc


class MolpSolver(metaclass=abc.ABCMeta):

    """Implements abstract MOLP solver"""

    def __init__(self, model):
        self._model = model

    @abc.abstractmethod
    def solve(self):
        """Solves the molp problem and returns the nondomminated frontier"""
