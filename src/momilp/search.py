"""Implements the search space elements and models"""

import abc
import copy
from enum import Enum
from gurobipy import Var
import math
import operator

from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, RayInTwoDimension, \
    FrontierEdgeInTwoDimension, FrontierInTwoDimension, FrontierSolution, LowerBoundInTwoDimension, \
    OptimizationStatus, Point, PointInTwoDimension, SearchRegionInTwoDimension, SearchProblemResult, SliceProblemResult
from src.molp.dichotomic_search.solver import BolpDichotomicSearchWithGurobiSolver
from src.momilp.utilities import ConstraintGenerationUtilities, ModelQueryUtilities, PointComparisonUtilities, \
    SearchUtilities


class Problem(metaclass=abc.ABCMeta):

    """Implements abstact problem class"""

    _SUPPORTED_SEARCH_REGION_NUM_DIMENSIONS = [2]
    _TABU_CONSTRAINT_NAME_TEMPLATE = "tabu_{idx}"
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

    def _add_tabu_constraint(self, y_bars):
        """Adds the tabu-constraints to the model for the given integer vectors"""
        momilp_model = self._momilp_model
        for index, y_bar in enumerate(y_bars):
            constraint_name = Problem._TABU_CONSTRAINT_NAME_TEMPLATE.format(idx=index)            
            ConstraintGenerationUtilities.create_tabu_constraint(momilp_model, constraint_name, y_bar)

    def _remove_region_defining_constraints(self):
        """Removes all the region defining constraints from the model"""
        momilp_model = self._momilp_model
        region_defining_constraint_names = momilp_model.region_defining_constraint_names()
        momilp_model.remove_constraint(region_defining_constraint_names)

    def _remove_tabu_constraints(self):
        """Removes the tabu-constraints from the model"""
        momilp_model = self._momilp_model
        tabu_constraint_names = momilp_model.tabu_constraint_names()
        momilp_model.remove_constraint(tabu_constraint_names)

    def _reset_model(self):
        """Removes all the previous tabu and region-defining constraints from the model"""
        self._remove_tabu_constraints()
        self._remove_region_defining_constraints()

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
    def update_problem(self, *args, **kwargs):
        """Updates the problem"""

    @abc.abstractmethod
    def solve(self):
        """Solves the problem"""


class SearchProblem(Problem):

    """Implements search problem to find nondominated points"""

    def __init__(self, momilp_model):
        super(SearchProblem, self).__init__(momilp_model)
        self._tabu_y_bars = []
        self._region = None
        self._relaxed_problem_result = None
        self._result = None
        self._solved_milp = False

    def _solve_relaxed_problem(self):
        """Solves the LP relaxation of the problem"""
        momilp_model = self._momilp_model
        momilp_model.relax()
        momilp_model.solve()
        self._relaxed_problem_result = ModelQueryUtilities.query_optimal_solution(
            momilp_model.problem(), momilp_model.y(), round_integer_vector_values=False)
        momilp_model.unrelax()

    def clear_result(self):
        """Clears the result"""
        self._result = None

    def num_tabu_constraints(self):
        """Returns the number of tabu-constraints"""
        return len(self._tabu_y_bars)

    def region(self):
        """Returns the last added search region"""
        return self._region

    def relaxed_problem_result(self):
        return self._relaxed_problem_result

    def result(self):
        return self._result

    def solve(self, solve_relaxed_problem=True):
        # the model has to be reset since the same model is copied over many search regions
        self._reset_model()
        if self._region:
            self._add_region_defining_constraints_in_two_dimension(self._region)
        self._add_tabu_constraint(self._tabu_y_bars)
        momilp_model = self._momilp_model
        if solve_relaxed_problem:
            self._solved_milp = False
            self._solve_relaxed_problem()
        if self._relaxed_problem_result.status() == OptimizationStatus.INFEASIBLE:
            point_solution = None
            self._result = SearchProblemResult(point_solution, OptimizationStatus.INFEASIBLE)
            return self._result
        momilp_model.solve()
        self._solved_milp = True
        self._result = ModelQueryUtilities.query_optimal_solution(momilp_model.problem(), momilp_model.y())
        return self._result

    def solved_milp(self):
        """Returns True if an MILP model had to be solved, otherwise False"""
        return self._solved_milp

    def tabu_y_bars(self):
        """Returns the tabu y_bars"""
        return self._tabu_y_bars

    def update_problem(self, keep_previous_tabu_constraints=False, region=None, tabu_y_bars=None):
        if not keep_previous_tabu_constraints:
            self._tabu_y_bars = []
        tabu_y_bars = tabu_y_bars or []
        self._tabu_y_bars.extend(tabu_y_bars)
        if region:
            SearchProblem._validate_search_region(region)
            self._region = region

    def update_region(self, region, clear_result=False):
        """Updates the region"""
        SearchProblem._validate_search_region(region)
        self._region = region
        if clear_result:
            self._result = None

    def update_result(self, result):
        """Updates the result"""
        self._result = result


class SearchSpace:

    """Implements search space for the momilp problem"""

    def __init__(
            self, primary_criterion_index, projected_space_criterion_index_2_criterion_index, dimension=3):
        assert primary_criterion_index not in projected_space_criterion_index_2_criterion_index.values(), \
            "the primary criterion cannot be a projected space criterion"
        assert dimension == 3, "only three dimensional search spaces are supported currently"
        self._dimension = dimension
        self._primary_criterion_index = primary_criterion_index
        # NOTE: Search problem are always sorted based on the value of projected space criterion at index 0
        self._search_problems = []

    def add_search_problem(self, search_problem, index=None):
        """Add the search problem to the search problems in the search space"""
        if index is None:
            self._search_problems.append(search_problem)
        else:
            self._search_problems.insert(index, search_problem)

    @staticmethod
    def couple_search_problems(search_problems):
        """Couples the search regions in a way that the coupled search regions are also convex, returns the coupled 
        search problems"""
        coupled_search_problems = []
        for search_problem in search_problems:
            if len(coupled_search_problems) == 0:
                coupled_search_problems.append(search_problem)
                continue
            search_problem_ = SearchSpace.couple_two_search_problem(coupled_search_problems[-1], search_problem)
            if search_problem_ is None:
                coupled_search_problems.append(search_problem)
                continue
            coupled_search_problems[-1] = search_problem_
        return coupled_search_problems
    
    @staticmethod
    def couple_two_search_problem(left_search_problem, right_search_problem):
        """Couples two neighbor search problems, and returns the coupled search problem. If the problems cannot be 
        coupled, returns None"""
        # couple the search problems only if neither of them has a result
        if left_search_problem.result() or right_search_problem.result():
            return
        momilp_model = left_search_problem.momilp_model()
        left_region = left_search_problem.region()
        right_region = right_search_problem.region()
        left_region_p, left_region_m = SearchUtilities.find_extreme_point_of_search_region_in_two_dimension(
            left_region, left_extreme=False)
        right_region_p, right_region_m = SearchUtilities.find_extreme_point_of_search_region_in_two_dimension(
            right_region, left_extreme=True)
        same_point = left_region_p == right_region_p
        convex = right_region_m >= left_region_m
        if not same_point or not convex:
            return
        if 0 < left_region_m < float("inf") and not math.isclose(left_region_m, right_region_m, rel_tol=1e-6):
            return
        lb_x = left_region.lower_bound().bounds()[0]
        to_couple_edges = []
        if 0 < left_region_m < float("inf"):
            assert left_region.edge() is not None, "there must be an edge in the left region"
            to_couple_edges.append(left_region.edge())
        if 0 < right_region_m < float("inf"):
            assert right_region.edge() is not None, "there must be an edge in the right region"
            to_couple_edges.append(right_region.edge())
        coupled_edge = EdgeInTwoDimension(to_couple_edges[0].left_point(), to_couple_edges[-1].right_point()) if \
            to_couple_edges else None
        lb_y = right_region.lower_bound().bounds()[1]
        coupled_bound = LowerBoundInTwoDimension([lb_x, lb_y])
        coupled_cone = ConvexConeInPositiveQuadrant(
            [left_region.cone().left_extreme_ray(), right_region.cone().right_extreme_ray()])
        coupled_region = SearchRegionInTwoDimension(
            left_region.x_obj_name(), left_region.y_obj_name(), coupled_cone, edge=coupled_edge, 
            lower_bound=coupled_bound, validate=False)
        results = []
        if left_search_problem.result():
            results.append(left_search_problem.result())
        if right_search_problem.result():
            results.append(right_search_problem.result())
        if len(results) == 0:
            better_result = None
        elif len(results) == 1:
            better_result = results[0]
        else:
            value_index_2_priority = momilp_model.objective_index_2_priority()
            better_result = results[0] if PointComparisonUtilities.compare_to(
                results[0].point_solution().point(), results[1].point_solution().point(), value_index_2_priority) >= 0 \
                else results[1]
        p = SearchProblem(momilp_model)
        tabu_y_bars = left_search_problem.tabu_y_bars() + right_search_problem.tabu_y_bars()
        p.update_problem(region=coupled_region, tabu_y_bars=tabu_y_bars)
        p.update_result(better_result)
        # print("left: ", left_search_problem.region(), left_search_problem.result().point_solution())
        # print("right: ", right_search_problem.region(), right_search_problem.result().point_solution())
        # print("coupled: ", p.region(), better_result.point_solution())
        return p

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


class SliceProblem(Problem):

    """Implements slice problem (in the projected space with n-1 dimension)"""

    _INVALID_INTEGER_VECTOR_ERROR_MESSAGE = "Failed to validate the problem for y='%s' and y_bar='%s'"

    def __init__(self, momilp_model, slice_prob_obj_index_2_original_prob_obj_index):
        super(SliceProblem, self).__init__(momilp_model)
        self._primary_objective_index = momilp_model.primary_objective_index()
        self._primary_objective_value = None
        self._region = None
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

    def _reset_model(self):
        """Removes all the previous tabu and region-defining constraints from the model, and restores the original 
        variable bounds"""
        self._remove_region_defining_constraints()
        self._remove_tabu_constraints()
        self._momilp_model.restore_original_bounds_of_integer_variables()

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
        self._reset_model()
        self._validate_integer_vector(self._y_bar)
        self._momilp_model.fix_integer_vector(self._y_bar)
        if self._region:
            self._add_region_defining_constraints_in_two_dimension(self._region)
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

    def update_problem(self, y_bar, primary_objective_value=0, region=None):
        self._primary_objective_value = primary_objective_value
        self._y_bar = y_bar
        if region:
            SliceProblem._validate_search_region(region)
            self._region = region
