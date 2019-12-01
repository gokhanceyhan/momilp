"""Implements the search space elements and models"""

import abc
from enum import Enum
from gurobipy import Var
from src.momilp.elements import SearchRegionInTwoDimension
from src.momilp.utilities import ConstraintGenerationUtilities, ModelQueryUtilities


class Problem(metaclass=abc.ABCMeta):

    """Implements abstact problem class"""

    _SUPPORTED_SEARCH_REGION_NUM_DIMENSIONS = [2]
    _UNSUPPORTED_SEARCH_REGION_DIMENSION_ERROR = \
        "the search region dimension of '%s' is not supported in the problem, the supported dimensions are '%s'"

    def __init__(self, momilp_model):
        self._momilp_model = momilp_model

    def _add_region_defining_constraints_in_two_dimension(self, region):
        """Adds the linear constraints to the model to restrict the feasible criterion space in 'x_obj_name' and 
        'y_obj_name' criteria to the specified region"""
        momilp_model = self._momilp_model
        x_var = momilp_model.Z()[region.x_obj_name()]
        y_var = momilp_model.Z()[region.y_obj_name()]
        ConstraintGenerationUtilities.create_constraints_for_cone_in_positive_quadrant(
            momilp_model, region.cone(), x_var, y_var, name=region.id())
        ConstraintGenerationUtilities.create_constraint_for_edge_in_two_dimension(
            momilp_model, region.edge(), x_var, y_var, name=region.id())
        ConstraintGenerationUtilities.create_constraints_for_lower_bound_in_two_dimension(
            momilp_model, region.lower_bound(), x_var, y_var, name=region.id())

    def _remove_region_defining_constraints(self):
        """Removes all the region defining constraints from the model"""
        momilp_model = self._momilp_model
        region_defining_constraint_names = momilp_model.region_defining_constraint_names()
        momilp_model.remove_constraint(region_defining_constraint_names)

    @staticmethod
    def _validate_search_region(region):
        """Validates the search region"""
        try:
            assert isinstance(region, SearchRegionInTwoDimension)
        except AssertionError as error:
            message = Problem._UNSUPPORTED_SEARCH_REGION_DIMENSION_ERROR % (
                region.dim(), Problem._SUPPORTED_SEARCH_REGION_NUM_DIMENSIONS)
            raise RuntimeError(message) from error
    
    def momilp_model(self):
        """Returns the momilp model"""
        return self._momilp_model

    @abc.abstractmethod
    def result(self):
        """Returns the problem result"""
    
    @abc.abstractmethod
    def update_model(self, **kwargs):
        """Updates the model"""

    @abc.abstractmethod
    def solve(self):
        """Solves the problem"""


class SearchProblem(Problem):

    """Implements search problem to find nondominated points"""

    _TABU_CONSTRAINT_NAME_TEMPLATE = "tabu_{idx}"

    def __init__(self, momilp_model):
        super(SearchProblem, self).__init__(momilp_model)
        self._num_tabu_constraints = 0
        self._result = None

    def _add_tabu_constraint(self, y_bars):
        """Adds the tabu-constraints to the model for the given integer vectors"""
        momilp_model = self._momilp_model
        binary_model = momilp_model.binary()
        for y_bar in y_bars:
            constraint_name = SearchProblem._TABU_CONSTRAINT_NAME_TEMPLATE.format(idx=self._num_tabu_constraints)
            if binary_model:
                ConstraintGenerationUtilities.create_binary_tabu_constraint(momilp_model, constraint_name, y_bar)
            else:
                ConstraintGenerationUtilities.create_integer_tabu_constraint(momilp_model, constraint_name, y_bar)
            self._num_tabu_constraints += 1

    def _remove_tabu_constraints(self):
        """Removes the tabu-constraints in the model"""
        momilp_model = self._momilp_model
        tabu_constraint_names = momilp_model.tabu_constraint_names()
        momilp_model.remove_constraint(tabu_constraint_names)
        self._num_tabu_constraints = 0

    def num_tabu_constraints(self):
        """Returns the number of tabu-constraints"""
        return self._num_tabu_constraints

    def result(self):
        self._result

    def solve(self):
        momilp_model = self._momilp_model
        momilp_model.solve()
        self._result = ModelQueryUtilities.query_optimal_solution(momilp_model.problem())
        return self._result

    def update_model(
            self, keep_previous_region_constraints=False, keep_previous_tabu_constraints=False, region=None, 
            tabu_y_bars=None):
        if not keep_previous_region_constraints:
            self._remove_region_defining_constraints()
        if region:
            SearchProblem._validate_search_region(region)
            self._add_region_defining_constraints_in_two_dimension(region)
        if not keep_previous_tabu_constraints:
            self._remove_tabu_constraints()
        if tabu_y_bars:
            self._add_tabu_constraint(tabu_y_bars)


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


class SliceProblem(Problem):

    """Implement slice problem"""

    _INVALID_INTEGER_VECTOR_ERROR_MESSAGE = "Failed to validate the problem for y='%s' and y_bar='%s'"

    def __init__(self, momilp_model):
        super(SliceProblem, self).__init__(momilp_model)
        self._result = None
        self._reduce()

    def _reduce(self):
        """Reduces the dimension of the momilp problem by dropping the primary objective function"""
        model = self._momilp_model.problem()
        primary_criterion_index = self._momilp_model.primary_criterion_index()
        obj_indices = [i for i in range(0, model.getAttr("NumObj")) if i != primary_criterion_index]
        obj_index_2_obj = {}
        obj_index_2_obj_name = {}
        for i in range(model.getAttr("NumObj")):
            obj_index_2_obj[i] =  model.getObjective(i)
            model.setParam("ObjNumber", i)
            obj_index_2_obj_name[i] = model.getAttr("ObjNName")
        filtered_objectives = [obj for i, obj in obj_index_2_obj.items() if i in obj_indices]
        num_obj = len(obj_indices)
        model.setAttr("NumObj", num_obj)
        for i in range(num_obj):
            model.setObjectiveN(filtered_objectives[i], i, name=obj_index_2_obj_name[i])

    def _validate_integer_vector(self, y_bar):
        """Validates the integer vector"""
        y = self._momilp_model.y()
        try:
            assert all([isinstance(y_bar_, int) for y_bar_ in y_bar])
            assert len(y_bar) == len(y)
            assert all([y_.getAttr("LB") <= y_bar_ <= y_.getAttr("UB") for y_, y_bar_ in zip(y, y_bar)])
        except AssertionError as error:
            message = SliceProblem._INVALID_INTEGER_VECTOR_ERROR_MESSAGE % (y, str(y_bar))
            raise RuntimeError(message) from error
    
    def result(self):
        return self._result

    def solve(self):
        pass

    def update_model(self, keep_previous_region_constraints=False, region=None, y_bar=None):
        if y_bar:
            self._validate_integer_vector(y_bar)
            self._momilp_model.fix_integer_vector(y_bar)
        if not keep_previous_region_constraints:
            self._remove_region_defining_constraints()
        if region:
            SliceProblem._validate_search_region(region)
            self._add_region_defining_constraints_in_two_dimension(region)
