"""Implements the search space elements and models"""

from enum import Enum
from gurobipy import Var
from src.momilp.elements import SearchRegionInTwoDimension
from src.momilp.utility import ConstraintGenerationUtilities


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

    _SUPPORTED_SEARCH_REGION_NUM_DIMENSIONS = [2]
    _UNSUPPORTED_SEARCH_REGION_DIMENSION_ERROR = \
        "the search region dimension of '%s' is not supported in the slice problem, the supported dimensions are '%s'"

    def __init__(self, model):
        self._model = model

    def _add_region_defining_constraints_in_two_dimension(self, region):
        """Adds the linear constraints to the model to restrict the feasible criterion space in 'x_obj_name' and 
        'y_obj_name' criteria to the specified region"""
        if not isinstance(region, SearchRegionInTwoDimension):
            message = SliceProblem._UNSUPPORTED_SEARCH_REGION_DIMENSION_ERROR % (
                region.dim(), SliceProblem._SUPPORTED_SEARCH_REGION_NUM_DIMENSIONS)
            raise ValueError(message)
        model = self._model
        x_var = model.Z()[region.x_obj_name()]
        y_var = model.Z()[region.y_obj_name()]
        ConstraintGenerationUtilities.create_constraints_for_cone_in_positive_quadrant(
            model, region.cone(), x_var, y_var, name=region.id())
        ConstraintGenerationUtilities.create_constraint_for_edge_in_two_dimension(
            model, region.edge(), x_var, y_var, name=region.id())
        ConstraintGenerationUtilities.create_constraints_for_lower_bound_in_two_dimension(
            model, region.lower_bound(), x_var, y_var, name=region.id())

    def _remove_region_defining_constraints(self):
        """Removes all the region defining constraints from the model"""
        model = self._model
        region_defining_constraint_names = model.region_defining_constraint_names()
        model.remove_constraint(region_defining_constraint_names)

    def _update_model(self, y_bar, region=None):
        """Updates the model"""
        self._model.fix_integer_vector(y_bar)
        if not region:
            return
        self._remove_region_defining_constraints()
        if isinstance(region, SearchRegionInTwoDimension):
            self._add_region_defining_constraints_in_two_dimension(region)

    def _validate(self, y_bar):
        """Validates the slice problem"""
        assert all([isinstance(y_bar_, int) for y_bar_ in y_bar])
        y = self._model.int_vars()
        assert all([isinstance(y_, Var) for y_ in y])
        assert len(y_bar) == len(y)
        assert all([y_.getAttr("LB") <= y_bar_ <= y_.getAttr("UB") for y_, y_bar_ in zip(y, y_bar)])

    def solve(self, y_bar, region=None):
        """Solves the slice problem for the given integer vector and return the nondominated frontier"""
        try:
            self._validate(y_bar)
        except AssertionError as error:
            message = "Failed to validate the slice problem for y='%s' and y_bar='%s'" % (
                self._model.int_vars(), str(y_bar))
            raise RuntimeError(message) from error
        self._update_model(y_bar, region)
        self._model.solve()
