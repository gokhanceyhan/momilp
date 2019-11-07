"""Implements the search space elements and models"""

from enum import Enum

class SearchProblem:

    """Implements search problem to find nondominated points"""

    _DEFAULT_DIMEMSION = 3

    def __init__(self, model, region, dimension=None, tabu_constraints=None):
        self._dimension = dimension or SearchProblem._DEFAULT_DIMEMSION
        self._frontier = None
        self._model = model
        self._region = region
        self._tabu_constraints = tabu_constraints or []
        self._y = None
        self._validate()

    def _validate(self):
        """Validates the search problem in three dimension"""

    def solve(self):
        """Solves the search problem and returns the search result"""


class SearchResult:

    """Implements search result"""

    def __init__(self, status, y_opt=None, z_opt=None):
        self._status = status
        self._y_opt = y_opt
        self._z_opt = z_opt


class SearchSpace:

    """Implements search space for the momilp problem"""

    _DEFAULT_DIMEMSION = 3

    def __init__(self, dimension=None, primary_criterion_index=1):
        self._dimension = SearchSpace._DEFAULT_DIMEMSION
        self._primary_criterion_index = primary_criterion_index
        self._search_problems = []
        self._initialize()

    def _initialize(self):
        """Initializes the search space"""

    def update(self):
        """Updates the search space"""


class SearchStatus(Enum):

    """Implements search status"""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    OPTIMAL = "optimal"


class SliceProblem:

    """Implement slice problem"""

    _DEFAULT_DIMEMSION = 3

    def __init__(self, model, dimension=None):
        self._dimension = dimension or SliceProblem._DEFAULT_DIMEMSION
        self._model = model

    def _update_model(self, y_bar, region=None)
        """Updates the model"""
        pass

    def _validate(self, y_bar, region=None):
        """Validates the slice problem"""
        pass

    def solve(self, y_bar, region=None):
        """Solves the slice problem for the given integer vector and return the nondominated frontier"""
        pass