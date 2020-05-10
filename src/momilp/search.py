"""Implements the search space elements and models"""

import abc
import copy
from enum import Enum
from gurobipy import Var
import math
import operator

from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, RayInTwoDimension, \
    FrontierEdgeInTwoDimension, FrontierInTwoDimension, FrontierSolution, LineInTwoDimension, \
    LowerBoundInTwoDimension, OptimizationStatus, Point, PointInTwoDimension, SearchRegionInTwoDimension, \
    SearchProblemResult, SliceProblemResult
from src.molp.dichotomic_search.solver import BolpDichotomicSearchWithGurobiSolver
from src.momilp.dominance import DominanceRules
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
        self._candidate_results = []
        self._region = None
        self._relaxed_problem_result = None
        self._result = None
        self._solved_milp = False
        self._tabu_y_bars = []

    def _solve_relaxed_problem(self):
        """Solves the LP relaxation of the problem"""
        momilp_model = self._momilp_model
        momilp_model.relax()
        momilp_model.solve()
        self._relaxed_problem_result = ModelQueryUtilities.query_optimal_solution(
            momilp_model.problem(), momilp_model.y(), round_integer_vector_values=False)
        momilp_model.unrelax()

    def add_candidate_result(self, result):
        """Adds the result to the candidate results"""
        self._candidate_results.append(result)

    def candidate_results(self):
        """Returns the candidate results"""
        return self._candidate_results

    def clear_candidate_result(self, index=0):
        """Removes the candidate result at the specified index"""    
        del self._candidate_results[index]

    def clear_result(self):
        """Clears the result"""
        self._result = None

    def is_feasible(self):
        """Returns True if the problem is feasible, False othwerwise"""
        return self._result.status() in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE] if self._result else \
            False

    def next_candidate_result(self):
        """Returns the next candidate result"""
        return self._candidate_results[0] if self._candidate_results else None

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

    def set_candidate_results(self, candidate_results):
        """Sets the candidate results"""
        self._candidate_results = candidate_results

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
        result = None
        if self._relaxed_problem_result.status() == OptimizationStatus.INFEASIBLE:
            point_solution = None
            result = SearchProblemResult(point_solution, OptimizationStatus.INFEASIBLE)
        else:
            momilp_model.solve()
            self._solved_milp = True
            result = ModelQueryUtilities.query_optimal_solution(momilp_model.problem(), momilp_model.y())
        candidate_result = self.next_candidate_result()
        if not candidate_result:
            self._result = result
            return self._result
        if result.status() not in [OptimizationStatus.FEASIBLE, OptimizationStatus.OPTIMAL]:
            self._result = candidate_result
            self.clear_candidate_result()
            return self._result
        value_index_2_priority = momilp_model.objective_index_2_priority()
        assert result.point_solution() and candidate_result.point_solution(), "this should not happen"
        if PointComparisonUtilities.compare_to(
                result.point_solution().point(), candidate_result.point_solution().point(), 
                value_index_2_priority) >= 0:
            self._result = result
        else:
            self._result = candidate_result
            self.clear_candidate_result()
            # add the result to the candidate results and update the tabu y-bars
            self._tabu_y_bars.append(result.point_solution().y_bar())
            self._candidate_results = SearchUtilities.sort_search_problem_results(
                self._candidate_results + [result], value_index_2_priority)
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

    def _couple_two_search_problem(self, left_search_problem, right_search_problem):
        """Couples two neighbor search problems, and returns the coupled search problem. If the problems cannot be 
        coupled, returns None"""
        momilp_model = left_search_problem.momilp_model()
        left_region = left_search_problem.region()
        right_region = right_search_problem.region()
        left_region_p, left_region_m = SearchUtilities.find_extreme_point_of_search_region_in_two_dimension(
            left_region, left_extreme=False)
        right_region_p, right_region_m = SearchUtilities.find_extreme_point_of_search_region_in_two_dimension(
            right_region, left_extreme=True)
        # continuity of the boundary with the dominated region
        same_point = left_region_p == right_region_p
        # convexity of the coupled region can only be achieved if the following condition is True.
        convex = right_region_m >= left_region_m
        if not same_point or not convex:
            return
        # construct the lower bound of the coupled region
        lb_x = left_region.lower_bound().bounds()[0]
        lb_y = right_region.lower_bound().bounds()[1]
        coupled_bound = LowerBoundInTwoDimension([lb_x, lb_y])
        coupled_edge = None
        if left_region.edge() and right_region.edge():
            coupled_edge = EdgeInTwoDimension(left_region.edge().left_point(), right_region.edge().right_point())
        elif left_region.edge():
            # we need to extend the edge to the right region as well
            line = LineInTwoDimension(left_region.edge().normal_vector(), left_region.edge().right_point())
            right_point = SearchUtilities.find_intersection_of_ray_and_line_in_two_dimension(
                line, right_region.cone().right_extreme_ray())
            coupled_edge = EdgeInTwoDimension(left_region.edge().left_point(), right_point)
        elif right_region.edge():
            # we need to extend the edge to the left region as well
            line = LineInTwoDimension(right_region.edge().normal_vector(), right_region.edge().left_point())
            left_point = SearchUtilities.find_intersection_of_ray_and_line_in_two_dimension(
                line, left_region.cone().left_extreme_ray())
            coupled_edge = EdgeInTwoDimension(left_point, right_region.edge().right_point())
        coupled_cone = ConvexConeInPositiveQuadrant(
            [left_region.cone().left_extreme_ray(), right_region.cone().right_extreme_ray()])
        coupled_region = SearchRegionInTwoDimension(
            left_region.x_obj_name(), left_region.y_obj_name(), coupled_cone, edge=coupled_edge, 
            lower_bound=coupled_bound)
        p = SearchProblem(momilp_model)
        results = []
        # couple the tabu integer vectors
        tabu_y_bars = [tuple(y_bar) for y_bar in left_search_problem.tabu_y_bars() + right_search_problem.tabu_y_bars()]
        if left_search_problem.result():
            tabu_y_bars.append(tuple(left_search_problem.result().point_solution().y_bar()))
            if left_search_problem.is_feasible():
                results.append(left_search_problem.result())
        if right_search_problem.result():
            tabu_y_bars.append(tuple(right_search_problem.result().point_solution().y_bar()))
            if right_search_problem.is_feasible():
                results.append(right_search_problem.result())
        results.extend(left_search_problem.candidate_results() + right_search_problem.candidate_results())
        tabu_y_bars = [list(y_bar) for y_bar in set(tabu_y_bars)]
        p.update_problem(region=coupled_region, tabu_y_bars=tabu_y_bars)
        # collect all the results in the coupled regions and select the best one
        value_index_2_priority = momilp_model.objective_index_2_priority()
        results = SearchUtilities.sort_search_problem_results(results, value_index_2_priority)
        best_result = results[0] if results else None
        candidate_results = results[1:] if results else []
        # eliminate candidate results that are dominated
        relatively_nd_candidate_results = []
        for candidate_result in candidate_results:
            point = candidate_result.point_solution().point()
            if DominanceRules.PointToPoint.dominated(point, best_result.point_solution().point()):
                continue
            relatively_nd_candidate_results.append(candidate_result)
        # set the best result only if both search problems have a result. Otherwise, it is possible that a better 
        # result can be found in the region with no result.
        if left_search_problem.result() and right_search_problem.result():
            p.update_result(best_result)
        elif best_result:
            p.add_candidate_result(best_result)
        for c in relatively_nd_candidate_results:
            p.add_candidate_result(c)
        return p

    def add_search_problem(self, search_problem, index=None):
        """Add the search problem to the search problems in the search space"""
        if index is None:
            self._search_problems.append(search_problem)
        else:
            self._search_problems.insert(index, search_problem)

    def couple_search_problems(self):
        """Couples the search regions in a way that the coupled search regions are also convex"""
        coupled_search_problems = []
        for search_problem in self._search_problems:
            if len(coupled_search_problems) == 0:
                coupled_search_problems.append(search_problem)
                continue
            search_problem_ = self._couple_two_search_problem(coupled_search_problems[-1], search_problem)
            if search_problem_ is None:
                coupled_search_problems.append(search_problem)
                continue
            coupled_search_problems[-1] = search_problem_
        self._search_problems[:] = coupled_search_problems

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

    def __init__(self, momilp_model, slice_prob_obj_index_2_original_prob_obj_index, dichotomic_search_rel_tol=1e-6):
        super(SliceProblem, self).__init__(momilp_model)
        self._dichotomic_search_rel_tol = dichotomic_search_rel_tol
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
        solver = BolpDichotomicSearchWithGurobiSolver(model, obj_rel_tol=self._dichotomic_search_rel_tol)
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
