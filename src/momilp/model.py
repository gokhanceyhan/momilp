"""Implements the momilp model"""

import abc

class AbstractModel(with_metaclass=abc.ABCMeta):

    """Implements the abstract class for the momilp model"""

    @abc.abstractmethod
    def add_constraint(self, id, lhs, rhs, sense):
        """Adds constraint to the problem"""

    @abc.abstractmethod
    def copy(self):
        """Creates and returns a deep copy of the model"""

    @abc.abstractmethod
    def fix_integer_vector(self, y_bar):
        """Creates the constraint of 'y = y_bar'"""

    @abc.abstractmethod
    def read_lp_file(self, file_path):
        """Reads the problem file in the '.lp' format"""

    @abc.abstractmethod
    def solve(self):
        """Solves the model"""

    @abc.abstractmethod
    def update_constraint(self, id):
        """Updates the constraint with the specified id"""

    @abc.abstractmethod
    def validate(self):
        """Validates the problem"""


class GurobiModel(AbstractModel):

    """Implements the model by using Gurobi solver"""

    def __init__(self):
        self._solver = None
