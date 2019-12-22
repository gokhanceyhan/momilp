"""Implements the search space elements and models"""

import abc
import copy
from enum import Enum
from gurobipy import Var
import operator
from src.momilp.dominance import DominanceRules
from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, RayInTwoDimension, \
    FrontierEdgeInTwoDimension, FrontierInTwoDimension, FrontierSolution, Point, PointInTwoDimension, \
    SearchRegionInTwoDimension, SliceProblemResult
from src.molp.dichotomic_search.solver import BolpDichotomicSearchWithGurobiSolver
from src.momilp.utilities import ConstraintGenerationUtilities, ModelQueryUtilities, PointComparisonUtilities


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
        if region.edge():
            ConstraintGenerationUtilities.create_constraint_for_edge_in_two_dimension(
                momilp_model, region.edge(), x_var, y_var, name=region.id())
        if region.lower_bound():
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
    
    def copy(self):
        """Creates and returns a deep copy of the object"""
        return copy.deepcopy(self)

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
        self._tabu_y_bars = []
        self._region = None
        self._result = None

    def _add_tabu_constraint(self, y_bars):
        """Adds the tabu-constraints to the model for the given integer vectors"""
        momilp_model = self._momilp_model
        binary_model = momilp_model.binary()
        for y_bar in y_bars:
            constraint_name = SearchProblem._TABU_CONSTRAINT_NAME_TEMPLATE.format(idx=len(self._tabu_y_bars))
            if binary_model:
                ConstraintGenerationUtilities.create_binary_tabu_constraint(momilp_model, constraint_name, y_bar)
            else:
                ConstraintGenerationUtilities.create_integer_tabu_constraint(momilp_model, constraint_name, y_bar)
            self._tabu_y_bars.append(y_bar)

    def _remove_tabu_constraints(self):
        """Removes the tabu-constraints in the model"""
        momilp_model = self._momilp_model
        tabu_constraint_names = momilp_model.tabu_constraint_names()
        momilp_model.remove_constraint(tabu_constraint_names)
        del self._tabu_y_bars[:]

    def num_tabu_constraints(self):
        """Returns the number of tabu-constraints"""
        return len(self._tabu_y_bars)

    def region(self):
        """Returns the last added search region"""
        return self._region

    def result(self):
        return self._result

    def solve(self):
        momilp_model = self._momilp_model
        momilp_model.solve()
        self._result = ModelQueryUtilities.query_optimal_solution(momilp_model.problem())
        return self._result

    def tabu_y_bars(self):
        """Returns the tabu y_bars"""
        return self._tabu_y_bars

    def update_model(
            self, keep_previous_region_constraints=False, keep_previous_tabu_constraints=False, region=None, 
            tabu_y_bars=None):
        if not keep_previous_region_constraints:
            self._remove_region_defining_constraints()
        if region:
            SearchProblem._validate_search_region(region)
            self._add_region_defining_constraints_in_two_dimension(region)
            self._region = region
        if not keep_previous_tabu_constraints:
            self._remove_tabu_constraints()
        if tabu_y_bars:
            self._add_tabu_constraint(tabu_y_bars)


class SearchSpace:

    """Implements search space for the momilp problem"""

    def __init__(
            self, primary_criterion_index, projected_space_criterion_index_2_criterion_index, dimension=3):
        assert primary_criterion_index not in projected_space_criterion_index_2_criterion_index.values(), \
            "the primary criterion cannot be a projected space criterion"
        assert dimension == 3, "only three dimensional search spaces are supported currently"
        self._dimension = dimension
        self._primary_criterion_index = primary_criterion_index
        self._projected_space_criterion_index_2_criterion_index = projected_space_criterion_index_2_criterion_index
        # NOTE: Search problem are always sorted based on the value of projected space criterion at index 0
        self._search_problems = []

    def add_search_problem(self, search_problem, index=None):
        """Add the search problem to the search problems in the search space"""
        if index is None:
            self._search_problems.append(search_problem)
        else:
            self._search_problems.insert(index, search_problem)

    def delete_search_problem(self, index=-1):
        """Deletes the search problem at the given index"""
        assert self._search_problems
        del self._search_problems[index]

    def search_problems(self):
        """Returns the search problems in the search space"""
        return self._search_problems

    def set_search_problems(self, search_problems):
        """Sets the search problems"""
        self._search_problems = search_problems

    def update_lower_bounds(self, reference_point, selected_search_problem_index, delta=0.0):
        """Updates the lower bounds of the search problems to eliminate the dominated regions by the reference point 
        and, solves the search problems again if their point solutions are dominated"""
        for index, search_problem in enumerate(self._search_problems):
            if index == selected_search_problem_index:
                continue
            update_bound_index = 0 if index > selected_search_problem_index else 1
            reference_point_value = reference_point.values()[
                self._projected_space_criterion_index_2_criterion_index[update_bound_index]]
            region = search_problem.region()
            lb = region.lower_bound().bounds()
            lb[update_bound_index] = max(lb[update_bound_index], reference_point_value + delta)
            point_solution = search_problem.result().point_solution()            
            if DominanceRules.PointToPoint.dominated(point_solution.point(), reference_point):
                search_problem.update_model(
                    region=region, keep_previous_tabu_constraints=True, tabu_y_bars=[point_solution.y_bar()])
                search_problem.solve()

class SliceProblem(Problem):

    """Implements slice problem (in the projected space with n-1 dimension)"""

    _INVALID_INTEGER_VECTOR_ERROR_MESSAGE = "Failed to validate the problem for y='%s' and y_bar='%s'"

    def __init__(self, momilp_model, slice_prob_obj_index_2_original_prob_obj_index):
        super(SliceProblem, self).__init__(momilp_model)
        self._primary_objective_index = momilp_model.primary_objective_index()
        self._primary_objective_value = None
        self._result = None
        self._slice_prob_obj_index_2_original_prob_obj_index = slice_prob_obj_index_2_original_prob_obj_index
        self._y_bar = None
        self._initialize()
    
    def _calculate_ideal_point(self, frontier_solution):
        """Returns the ideal point"""
        criterion_index_2_max_value = {i: v for i, v in enumerate(frontier_solution.frontier().point().values())} if \
            frontier_solution.frontier().point() else {}
        for edge in frontier_solution.frontier().edges():
            for index, value in enumerate(edge.start_point().values()):
                if criterion_index_2_max_value.get(index) is None or value > criterion_index_2_max_value[index]:
                    criterion_index_2_max_value[index] = value
            for index, value in enumerate(edge.end_point().values()):
                if criterion_index_2_max_value.get(index) is None or value > criterion_index_2_max_value[index]:
                    criterion_index_2_max_value[index] = value
        ideal_point_obj_index_2_value = {
            self._slice_prob_obj_index_2_original_prob_obj_index[i]: value for i, value in 
            criterion_index_2_max_value.items()}
        ideal_point_obj_index_2_value[self._primary_objective_index] = self._primary_objective_value
        values = [item[1] for item in sorted(ideal_point_obj_index_2_value.items(), key=operator.itemgetter(0))]
        return Point(values)

    def _initialize(self):
        """Initializes the slice problem"""
        self._reduce_dimension()
        self._relax_integrality()

    def _reduce_dimension(self):
        """Reduces the dimension of the momilp problem by dropping the primary objective function"""
        model = self._momilp_model.problem()
        primary_objective_index = self._primary_objective_index
        obj_index_2_obj = {}
        obj_index_2_obj_name = self._momilp_model.objective_index_2_name()
        for i in range(model.getAttr("NumObj")):
            obj_index_2_obj[i] =  model.getObjective(i)
        # first remove the objective equation of the primary object from the constraints
        self._momilp_model.remove_constraint(obj_index_2_obj_name[primary_objective_index])
        # update the objectives
        model.setAttr("NumObj", len(self._slice_prob_obj_index_2_original_prob_obj_index.keys()))
        for new_obj_index, old_obj_index in self._slice_prob_obj_index_2_original_prob_obj_index.items():
            model.setObjectiveN(
                obj_index_2_obj[old_obj_index], new_obj_index, name=obj_index_2_obj_name[old_obj_index])
        model.update()

    def _relax_integrality(self):
        """Relaxes the integrality constraints in the model"""
        self._momilp_model.relax()

    def _validate_integer_vector(self, y_bar):
        """Validates the integer vector"""
        y = self._momilp_model.y()
        try:
            assert all([isinstance(y_bar_, int) or y_bar_.is_integer() for y_bar_ in y_bar])
            assert len(y_bar) == len(y)
            assert all([y_.getAttr("LB") <= y_bar_ <= y_.getAttr("UB") for y_, y_bar_ in zip(y, y_bar)])
        except BaseException as error:
            message = SliceProblem._INVALID_INTEGER_VECTOR_ERROR_MESSAGE % (y, str(y_bar))
            raise RuntimeError(message) from error

    def result(self):
        return self._result

    def slice_prob_obj_index_2_original_prob_obj_index(self):
        """Returns the slice problem objective index to original problem objective index"""
        return self._slice_prob_obj_index_2_original_prob_obj_index

    def solve(self):
        model = self._momilp_model.problem()
        solver = BolpDichotomicSearchWithGurobiSolver(model)
        solver.solve()
        points = solver.extreme_supported_nondominated_points()
        if len(points) == 1:
            frontier_solution = FrontierSolution(FrontierInTwoDimension(point=points[0]), self._y_bar)
        else:
            edges = []
            for index, point in enumerate(points):
                if len(points) > index + 1:
                    edges.append(
                        FrontierEdgeInTwoDimension(
                            left_point=point, right_point=points[index + 1], z3=self._primary_objective_value))
            frontier_solution = FrontierSolution(FrontierInTwoDimension(edges=edges), self._y_bar)
        ideal_point = self._calculate_ideal_point(frontier_solution)
        self._result = SliceProblemResult(frontier_solution, ideal_point)
        return self._result

    def update_model(self, y_bar, keep_previous_region_constraints=False, primary_objective_value=0, region=None):
        self._primary_objective_value = primary_objective_value
        self._y_bar = y_bar
        self._momilp_model.restore_original_bounds_of_integer_variables()
        self._validate_integer_vector(y_bar)
        self._momilp_model.fix_integer_vector(y_bar)
        if not keep_previous_region_constraints:
            self._remove_region_defining_constraints()
        if region:
            SliceProblem._validate_search_region(region)
            self._add_region_defining_constraints_in_two_dimension(region)
