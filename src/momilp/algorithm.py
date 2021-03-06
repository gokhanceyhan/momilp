"""Implements algorithm to solve the momilp"""

import abc
import copy
from enum import Enum
from gurobipy import read
import logging
import operator
import os
from time import time

from src.common.elements import ConvexConeInPositiveQuadrant, Edge, EdgeInTwoDimension, EdgeSolution, \
    RayInTwoDimension, FrontierInTwoDimension, FrontierSolution, LowerBoundInTwoDimension, OptimizationStatus, Point, \
    PointInTwoDimension, PointSolution, SearchRegionInTwoDimension, SearchProblemResult, SliceProblemResult
from src.momilp.dominance import DominanceRules, ModelBasedDominanceFilter
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SearchProblem, SearchSpace, SliceProblem
from src.momilp.state import Iteration, IterationStatistics, SolutionState, State
from src.momilp.utilities import point_on_ray_in_two_dimension, PointComparisonUtilities, SearchUtilities, \
    TypeConversionUtilities


class AbstractAlgorithm(metaclass=abc.ABCMeta):

    """Implements abstract class for the algorithm"""

    @abc.abstractmethod
    def errors(self):
        """Returns the errors"""
    
    @abc.abstractmethod
    def run(self):
        """Runs the algorithm"""


class AlgorithmType(Enum):

    """Implements algorithm type"""

    CONE_BASED_SEARCH = "cone-based search"


class AlgorithmFactory:

    """Implements algorithm factory to solve momilp"""

    _SUPPORTED_ALGORITHM_TYPES = [AlgorithmType.CONE_BASED_SEARCH]
    _SUPPORTED_NUM_OBJECTIVES = [3]
    _UNSUPPORTED_ALGORITHM_TYPE_ERROR_MESSAGE = \
        "the '{type!s}' algorithm type is not supported, select one of the '{supported_types!s}' types"
    _UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE = \
        "the '{num_obj!s}'-obj problem is not supported, select one of the '{supported_num_obj!s}' values"

    @staticmethod
    def _create_cone_based_search_algorithm(
            model_file, working_dir, dichotomic_search_rel_tol=1e-6, discrete_objective_indices=None, 
            explore_decision_space=False, max_num_iterations=None, obj_index_2_range=None, rel_coverage_gap=0.0, 
            search_model_params_file=None, search_num_threads=None, search_time_limit_in_seconds=None, 
            slice_model_params_file=None):
        """Creates and returns the cone-based search algorithm"""
        return ConeBasedSearchAlgorithm(
            model_file, working_dir, dichotomic_search_rel_tol=dichotomic_search_rel_tol, 
            discrete_objective_indices=discrete_objective_indices, explore_decision_space=explore_decision_space, 
            max_num_iterations=max_num_iterations, obj_index_2_range=obj_index_2_range, 
            rel_coverage_gap=rel_coverage_gap, search_model_params_file=search_model_params_file, 
            search_num_threads=search_num_threads, search_time_limit_in_seconds=search_time_limit_in_seconds, 
            slice_model_params_file=slice_model_params_file)

    @staticmethod
    def create(
            model_file, working_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            dichotomic_search_rel_tol=1e-6, discrete_objective_indices=None, explore_decision_space=False, 
            max_num_iterations=None, obj_index_2_range=None, rel_coverage_gap=0.0, search_model_params_file=None, 
            search_num_threads=None, search_time_limit_in_seconds=None, slice_model_params_file=None):
        """Creates an algorithm"""
        model = read(model_file)
        num_obj = model.num_obj
        if num_obj not in AlgorithmFactory._SUPPORTED_NUM_OBJECTIVES:
            error_message = AlgorithmFactory._UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE.format(
                num_obj=num_obj, supported_num_obj=AlgorithmFactory._SUPPORTED_NUM_OBJECTIVES)
            raise ValueError(error_message)
        if algorithm_type not in AlgorithmFactory._SUPPORTED_ALGORITHM_TYPES:
            error_message = AlgorithmFactory._UNSUPPORTED_ALGORITHM_TYPE_ERROR_MESSAGE.format(
                type=algorithm_type.value, 
                supported_types=[type_.value for type_ in AlgorithmFactory._SUPPORTED_ALGORITHM_TYPES])
            raise ValueError(error_message)
        # We need to use the model file instead of model itself in order to create different model objects
        if algorithm_type == AlgorithmType.CONE_BASED_SEARCH:
            return AlgorithmFactory._create_cone_based_search_algorithm(
                model_file, working_dir, dichotomic_search_rel_tol=dichotomic_search_rel_tol, 
                discrete_objective_indices=discrete_objective_indices, explore_decision_space=explore_decision_space, 
                max_num_iterations=max_num_iterations, obj_index_2_range=obj_index_2_range, 
                rel_coverage_gap=rel_coverage_gap, search_num_threads=search_num_threads, 
                search_time_limit_in_seconds=search_time_limit_in_seconds, 
                search_model_params_file=search_model_params_file, slice_model_params_file=slice_model_params_file)


class ConeBasedSearchAlgorithm(AbstractAlgorithm):

    """Implements the cone-based search algorithm"""

    _ALPHA = 1e-5
    _STARTING_ITERATION_INDEX = 0

    def __init__(
            self, model_file, working_dir, dichotomic_search_rel_tol=1e-6, discrete_objective_indices=None, 
            explore_decision_space=False, max_num_iterations=None, obj_index_2_range=None, rel_coverage_gap=0.0, 
            search_model_params_file=None, search_num_threads=None, search_time_limit_in_seconds=None, 
            slice_model_params_file=None):
        self._dichotomic_search_rel_tol = dichotomic_search_rel_tol
        self._discrete_objective_indices = discrete_objective_indices or []
        self._dominance_filter = None
        self._elapsed_time_in_seconds_for_search_problem = 0
        self._elapsed_time_in_seconds_for_slice_problem = 0
        self._errors = []
        self._explore_decision_space = explore_decision_space
        self._max_num_iterations = max_num_iterations
        self._model_file = model_file
        self._momilp_model = None
        self._num_milp_solved = 0
        self._objective_index_2_priority = {}
        self._obj_index_2_range = obj_index_2_range or {}
        self._primary_objective_index = None
        self._projected_space_criterion_index_2_criterion_index = {}
        self._rel_coverage_gap = rel_coverage_gap
        self._search_model_params_file = search_model_params_file
        self._search_num_threads = search_num_threads
        self._search_time_limit_in_seconds = search_time_limit_in_seconds
        self._slice_model_params_file = slice_model_params_file
        self._state = None
        self._x_obj_name = None
        self._y_obj_name = None
        self._working_dir = working_dir
        self._initialize()
        
    def _add_iteration(self, iteration_index, iteration_time_in_seconds, selected_point_solution):
        """Creates and add the completed iteration to the list of iterations in the state"""
        state = self._state
        num_milp_solved = self._num_milp_solved
        elapsed_time_in_seconds_for_search_problem = self._elapsed_time_in_seconds_for_search_problem
        elapsed_time_in_seconds_for_slice_problem = self._elapsed_time_in_seconds_for_slice_problem
        num_search_problems = len(state.search_space().search_problems())
        avg_num_tabu_constraints = sum(sp.num_tabu_constraints() for sp in state.search_space().search_problems()) / \
            num_search_problems if num_search_problems else 0.0
        statistics = IterationStatistics(
            avg_num_tabu_constraints, iteration_time_in_seconds, elapsed_time_in_seconds_for_search_problem, 
            elapsed_time_in_seconds_for_slice_problem, num_milp_solved, num_search_problems)
        iteration = Iteration(iteration_index, selected_point_solution, statistics)
        state.iterations().append(iteration)
        # reset the statistics of the algorithm instance
        self._num_milp_solved = 0
        self._elapsed_time_in_seconds_for_search_problem = 0
        self._elapsed_time_in_seconds_for_slice_problem = 0

    def _convert_edge_in_projected_space_to_edge_in_original_space(self, edge):
        """Returns an edge in the original search space from the edge in the projected search space"""
        additional_dim_2_value = {self._primary_objective_index: edge.z3()}
        return TypeConversionUtilities.edge_in_two_dimension_to_edge(
            additional_dim_2_value, edge, self._projected_space_criterion_index_2_criterion_index)

    def _create_momilp_model(self, params_file=None):
        """Creates and returns a momilp model"""
        return GurobiMomilpModel(
            self._model_file, discrete_objective_indices=self._discrete_objective_indices, 
            num_threads=self._search_num_threads, obj_index_2_range=self._obj_index_2_range, params_file=params_file, 
            time_limit_in_seconds=self._search_time_limit_in_seconds)

    def _create_positive_quadrant_convex_cone(self):
        """Returns a convex cone corresponding to the positive quadrant"""
        return ConvexConeInPositiveQuadrant(
            [RayInTwoDimension(90, PointInTwoDimension([0, 0])), RayInTwoDimension(0, PointInTwoDimension([0, 0]))])

    def _create_pseudo_search_region_in_two_dimension(self, frontier):
        """Creates a pseudo-search region in two dimension by using the given frontier
        
        The search region has a cone with extreme rays passing through the left-most and right-most extreme points, and 
        an edge defined with the left-most and right-most extreme points of the frontier
        
        NOTE: If the frontier is a singleton, or it consists of a single edge, then returns None."""
        if frontier.point():
            return
        edges = frontier.edges()
        if len(edges) == 1:
            return
        left_most_edge = edges[0]
        right_most_edge = edges[-1]
        left_extreme_ray = SearchUtilities.create_ray_in_two_dimension(
            PointInTwoDimension([0, 0]), left_most_edge.left_point())
        right_extreme_ray = SearchUtilities.create_ray_in_two_dimension(
            PointInTwoDimension([0, 0]), right_most_edge.right_point())
        cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
        edge = EdgeInTwoDimension(left_most_edge.left_point(), right_most_edge.right_point())
        return SearchRegionInTwoDimension(self._x_obj_name, self._y_obj_name, cone, edge=edge)

    @staticmethod
    def _filter_search_problems(problems, y_bar=None):
        """Filter the search problems based on the filtering conditions"""
        filtered_problems = [p for p in problems]
        if y_bar is not None:
            filtered_problems = [p for p in filtered_problems if p.result().point_solution().y_bar() == y_bar]
        return filtered_problems

    def _initialize(self):
        """Initializes the algorithm"""
        # set the primary objective index, objective priorities and projected space criterions
        momilp_model = self._create_momilp_model()
        self._primary_objective_index = momilp_model.primary_objective_index()
        self._objective_index_2_priority = momilp_model.objective_index_2_priority()
        filtered_obj_indices = [i for i in range(0, momilp_model.num_obj()) if i != self._primary_objective_index]
        for new_obj_index, old_obj_index in enumerate(filtered_obj_indices):
            self._projected_space_criterion_index_2_criterion_index[new_obj_index] = old_obj_index
        x_obj_index = self._projected_space_criterion_index_2_criterion_index[0]
        y_obj_index = self._projected_space_criterion_index_2_criterion_index[1]
        self._x_obj_name = momilp_model.objective_index_2_name()[x_obj_index]
        self._y_obj_name = momilp_model.objective_index_2_name()[y_obj_index]
        # NOTE: Use differemt instances of momilp model in search and slice problems
        self._momilp_model = momilp_model
        # create the initial search problem
        cone = self._create_positive_quadrant_convex_cone()
        region = SearchRegionInTwoDimension(self._x_obj_name, self._y_obj_name, cone)
        try:
            search_problem = SearchProblem(self._create_momilp_model(params_file=self._search_model_params_file))
        except BaseException as e:
            raise RuntimeError(
                "failed to create the search problem in the initialization of the cone-based search algorithm") from e
        search_problem.update_problem(region=region)
        # create a slice problem
        try:
            slice_problem = SliceProblem(
                self._create_momilp_model(params_file=self._slice_model_params_file), 
                self._projected_space_criterion_index_2_criterion_index, 
                dichotomic_search_rel_tol=self._dichotomic_search_rel_tol)
        except BaseException as e:
            raise RuntimeError(
                "failed to create the slice problem in the initialization of the cone-based search algorithm") from e
        # create the initial search space
        search_space = SearchSpace(
            self._primary_objective_index, self._projected_space_criterion_index_2_criterion_index)
        search_space.add_search_problem(search_problem)
        # create the initial solution state
        solution_state = SolutionState()
        self._state = State(search_space, slice_problem, solution_state=solution_state)
        # create the dominance filter that will be used to filter the dominated points of weakly nondominated sets
        self._dominance_filter = ModelBasedDominanceFilter(momilp_model.num_obj() - 1)

    @staticmethod
    def _log_iteration_status(**kwargs):
        """Logs the given arguments to the console"""
        for key, value in kwargs.items():
            logging.info("%s = %s" % (key, value))

    def _partition_search_problem(self, frontier, selected_search_problem, tol=1e-6):
        """Partitions the search problem and returns the new search problems"""
        # add the integer vector to the list of tabu integer vectors
        projected_space_criterion_indices = self._projected_space_criterion_index_2_criterion_index.values()
        selected_region = selected_search_problem.region()
        selected_point_solution = selected_search_problem.result().point_solution()
        y_bar = selected_point_solution.y_bar()
        tabu_y_bars = selected_search_problem.tabu_y_bars()
        if y_bar not in tabu_y_bars:
            tabu_y_bars.append(y_bar)
        # partition the selected search region
        child_search_regions = SearchUtilities.partition_search_region_in_two_dimension(frontier, selected_region)
        # test the child search regions for infeasibility
        pseudo_search_region = self._create_pseudo_search_region_in_two_dimension(frontier)
        pseudo_search_problem = None
        feasible_child_region_index = None
        if pseudo_search_region:
            pseudo_search_problem = SearchProblem(selected_search_problem.momilp_model())
            pseudo_search_problem.update_problem(region=pseudo_search_region, tabu_y_bars=tabu_y_bars)
            self._solve_search_problem(pseudo_search_problem)
            if pseudo_search_problem.is_feasible():
                if self._momilp_model.biobjective():
                    point = pseudo_search_problem.result().point_solution().point()
                    point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(
                        projected_space_criterion_indices, point)
                    ray_of_point = SearchUtilities.create_ray_in_two_dimension(
                        PointInTwoDimension([0, 0]), point_in_two_dimension)
                    # find the index of the child region where the point is found
                    for child_search_region_index, region in enumerate(child_search_regions):
                        if region.cone().left_extreme_ray().angle_in_degrees() < ray_of_point.angle_in_degrees() - tol:
                            continue
                        if ray_of_point.angle_in_degrees() < region.cone().right_extreme_ray().angle_in_degrees() - tol:
                            continue
                        feasible_child_region_index = child_search_region_index
                    assert feasible_child_region_index is not None, "this should not happen"
                    # the child regions on the right-hand-side cannot have a nd point
                    child_search_regions = child_search_regions[:feasible_child_region_index + 1]
            else:
                child_search_regions_ = []
                if not point_on_ray_in_two_dimension(
                        frontier.edges()[0].left_point(), selected_region.cone().left_extreme_ray()):
                    child_search_regions_.append(child_search_regions[0])
                if not self._momilp_model.biobjective() and not point_on_ray_in_two_dimension(
                        frontier.edges()[-1].right_point(), selected_region.cone().right_extreme_ray()):
                    child_search_regions_.append(child_search_regions[-1])
                child_search_regions = child_search_regions_
        # create the new search problems
        search_problems = []
        for child_index, region in enumerate(child_search_regions):
            search_problem = SearchProblem(selected_search_problem.momilp_model())
            tabu_y_bars_ = tabu_y_bars if self._explore_decision_space else []
            search_problem.update_problem(region=region, tabu_y_bars=tabu_y_bars_)
            # allocate the candidate results of the selected search problem to the child search problem
            for candidate_result in selected_search_problem.candidate_results():
                point = candidate_result.point_solution().point()
                point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(
                    projected_space_criterion_indices, point)
                if SearchUtilities.is_point_in_region_in_two_dimension(point_in_two_dimension, region):
                    search_problem.add_candidate_result(candidate_result)
            search_problems.append(search_problem)
            if child_index != feasible_child_region_index:
                continue
            if not pseudo_search_problem or not pseudo_search_problem.is_feasible():
                continue
            status = pseudo_search_problem.result().status()
            point_solution = pseudo_search_problem.result().point_solution()
            point = point_solution.point()
            point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(
                projected_space_criterion_indices, point)
            if not SearchUtilities.is_point_in_region_in_two_dimension(point_in_two_dimension, region):
                continue
            # set the already found point in pseudo problem in case it is not dominated by the frontier
            search_problem.update_result(SearchProblemResult(point_solution, status))
        return search_problems

    def _select_search_problem_and_index(self, search_problems):
        """Selects the search problem, returns with index"""
        selected_search_problem_and_index = None
        for index, search_problem in enumerate(search_problems):
            if not selected_search_problem_and_index:
                selected_search_problem_and_index = (search_problem, index)
            else:
                selected_search_problem = selected_search_problem_and_index[0]
                selected_point = selected_search_problem.result().point_solution().point()
                candiate_point = search_problem.result().point_solution().point()
                if PointComparisonUtilities.compare_to(
                        selected_point, candiate_point, self._objective_index_2_priority) < 0:
                    selected_search_problem_and_index = (search_problem, index)
        return selected_search_problem_and_index

    def _solve_search_problem(self, search_problem):
        """Solves the search problem"""
        start = time()
        result = search_problem.solve()
        if search_problem.solved_milp():
            self._num_milp_solved += 1
        end = time()
        self._elapsed_time_in_seconds_for_search_problem += end - start
        return result

    def _solve_search_problems(self, iteration_index, search_problems):
        """Solves and returns the search problems"""
        # search problems are sorted based on the regions from north-west to south-east in the x-y plane
        # x-axis: the criterion at index 0 of the projected space criteria
        # y-axis: the criterion at index 1 of the projected space criteria
        
        # idea: if a point in the left-hand-side region has a point, z^l, with z^l_x value larger than the z_^r_x value 
        # of the point in the right-hand-side region, z^r, then z^l dominated z^r.
        feasible_search_problems = []
        # solve the search problems
        for search_problem in search_problems:
            if search_problem.is_feasible():
                feasible_search_problems.append(search_problem)
                continue
            try:
                self._solve_search_problem(search_problem)
            except BaseException as e:
                self._errors.append(
                    "the problem '%s' failed with error '%s' in iteration '%s'" % (
                        id(search_problem), str(e), iteration_index))
                search_problem.momilp_model().write(os.path.join(self._working_dir, str(id(search_problem)) + ".lp"))
            else:
                if not search_problem.is_feasible():
                    continue
                feasible_search_problems.append(search_problem)
        search_problems[:] = [p for p in feasible_search_problems]
        return search_problems

    def _solve_slice_problem(self, iteration_index, region, selected_point_solution):
        """Solves the slice problem and returns the result"""
        # create a slice problem result from the selected solution. If slice problem is infeasible 
        # (which is supposed to be always feasible, but may be infeasible due to tolerances used), use this result
        start = time()
        projected_space_criterion_indices = self._projected_space_criterion_index_2_criterion_index.values()
        point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(
            projected_space_criterion_indices, selected_point_solution.point())
        frontier = FrontierInTwoDimension(point=point_in_two_dimension)
        frontier_solution = FrontierSolution(frontier, selected_point_solution.y_bar())
        ideal_point = selected_point_solution.point()
        result = SliceProblemResult(frontier_solution, ideal_point)
        if self._momilp_model.discrete_nondominated_set():
            end = time()
            self._elapsed_time_in_seconds_for_slice_problem = end - start
            return result
        slice_problem = self._state.slice_problem()
        slice_problem.update_problem(
            selected_point_solution.y_bar(), 
            primary_objective_value=selected_point_solution.point().values()[self._primary_objective_index], 
            region=region)
        try:
            result = slice_problem.solve()
        except BaseException as e:
            logging.info(
                "failed to solve the slice problem for point solution '%s' in region '%s' at iteration '%s', details: "
                "'%s'" % (selected_point_solution, region, iteration_index, str(e)))        
        end = time()
        self._elapsed_time_in_seconds_for_slice_problem = end - start
        return result

    def _update_lower_bounds(self, reference_point, search_problems, selected_search_problem_index):
        """Updates the lower bounds of the search problems to eliminate the dominated regions by the reference point"""
        for index, search_problem in enumerate(search_problems):
            if index == selected_search_problem_index:
                continue
            # since the search problem are indexed starting from the north-west region
            update_bound_index = 0 if index > selected_search_problem_index else 1
            update_criterion_index = self._projected_space_criterion_index_2_criterion_index[update_bound_index]
            reference_point_value = reference_point.values()[update_criterion_index]
            region = search_problem.region()
            lb = region.lower_bound().bounds()
            needs_update = lb[update_bound_index] < reference_point_value
            if needs_update:
                lb[update_bound_index] = reference_point_value
                region = SearchUtilities.create_search_region_in_two_dimension(
                    region.x_obj_name(), region.y_obj_name(), region.cone(), edge=region.edge(), 
                    lower_bound=LowerBoundInTwoDimension(lb), id_=region.id())
                search_problem.update_region(region)
            point_solution = search_problem.result().point_solution()
            if point_solution.point().values()[update_criterion_index] <= lb[update_bound_index]:
                search_problem.clear_result()
                tabu_y_bars = [point_solution.y_bar()] if self._explore_decision_space else []
                search_problem.update_problem(keep_previous_tabu_constraints=True, tabu_y_bars=tabu_y_bars)
            filtered_candidate_results = []
            for candidate_result in search_problem.candidate_results():
                point = candidate_result.point_solution().point()
                if point.values()[update_criterion_index] <= lb[update_bound_index]:
                    continue
                filtered_candidate_results.append(candidate_result)
            search_problem.set_candidate_results(filtered_candidate_results)

    def _update_state(self, selected_point_solution, frontier, iteration_index):
        """Updates the state"""
        state = self._state
        if iteration_index > ConeBasedSearchAlgorithm._STARTING_ITERATION_INDEX:
            # determine the status of the weakly nondominated points or edges generated in the previous iterations
            previous_selected_point_solution = state.iterations()[iteration_index - 1].selected_point_solution()
            if selected_point_solution.point().values()[self._primary_objective_index] < \
                    previous_selected_point_solution.point().values()[self._primary_objective_index]:
                state.solution_state().move_weakly_nondominated_to_nondominated(
                    constant_value_index=self._primary_objective_index)
            else:
                # we can apply the following tests to detect the nondominated points or edges since we know that 
                # the selected solution is the one having the highest value in the first criterion of the not-yet 
                # dominated projected space
                compared_criterion_index = self._projected_space_criterion_index_2_criterion_index[0]
                compared_criterion_value = selected_point_solution.point().values()[compared_criterion_index]
                state.solution_state().filter_weakly_nondominated_edges(
                    criterion_index=compared_criterion_index, criterion_value=compared_criterion_value)
                state.solution_state().filter_weakly_nondominated_points(
                    criterion_index=compared_criterion_index, criterion_value=compared_criterion_value)
                # the elements that are still in the weakly nondominated set will be tested for dominance
                state.solution_state().filter_dominated_points_and_edges(
                    self._dominance_filter, frontier, self._projected_space_criterion_index_2_criterion_index)
        # determine the status of the current frontier
        if frontier.singleton():
            # we need to check if the point is a weakly nondominated, but a dominated point. This is needed when the 
            # generated point is on the boundary of the search region
            dominated_by_prev_nd_point = DominanceRules.PointToPointSet.dominated(
                selected_point_solution.point(), 
                [nd_point.point() for nd_point in state.solution_state().nondominated_points()])
            dominated_by_prev_nd_edge = DominanceRules.PointToEdgeSet.dominated(
                selected_point_solution.point(), 
                [nd_edge.edge() for nd_edge in state.solution_state().nondominated_edges()], 
                edge_dimensions_with_constant_value=[self._primary_objective_index])
            if not dominated_by_prev_nd_point and not dominated_by_prev_nd_edge:
                state.solution_state().add_nondominated_point(selected_point_solution)
                state.solution_state().add_efficient_integer_vector(selected_point_solution.y_bar())
        else:
            # MOMILP_TO_DO: MOMILP-8: we need to check against to the nondominated edge set and eliminate the dominated 
            # ones
            edges = frontier.edges()
            # add the edges in reverse order since the generated points are sorted from right-to-left in two dimension
            for edge in edges[::-1]:
                solution_edge = self._convert_edge_in_projected_space_to_edge_in_original_space(edge)
                state.solution_state().add_weakly_nondominated_edge(
                    EdgeSolution(solution_edge, selected_point_solution.y_bar()), 
                    constant_value_index=self._primary_objective_index)

    def dominance_filter(self):
        """Returns the dominance filter"""
        return self._dominance_filter

    def errors(self):
        return self._errors

    def momilp_model(self):
        """Returns the momilp model"""
        return self._momilp_model

    def run(self, debug_mode=False):
        state = self._state
        search_space = state.search_space()
        search_problems = search_space.search_problems()
        rel_coverage_gap = 0.0 if self._explore_decision_space else self._rel_coverage_gap if \
            self._rel_coverage_gap > 0 else ConeBasedSearchAlgorithm._ALPHA
        iteration_index = ConeBasedSearchAlgorithm._STARTING_ITERATION_INDEX
        iteration_limit = (iteration_index + self._max_num_iterations - 1) if self._max_num_iterations is not None \
            else None 
        while search_problems and (iteration_limit is None or iteration_index <= iteration_limit):
            iteration_start_time = time()
            # search all of the regions and remove the infeasible ones
            search_problems = self._solve_search_problems(iteration_index, search_problems)
            if not search_problems:
                state.solution_state().move_weakly_nondominated_to_nondominated(
                    constant_value_index=self._primary_objective_index)
                break
            # select the next efficient integer vector
            selected_search_problem_and_index = self._select_search_problem_and_index(search_problems)
            selected_search_problem = selected_search_problem_and_index[0]
            selected_search_problem_index = selected_search_problem_and_index[1]
            selected_region = selected_search_problem.region()
            selected_point_solution = selected_search_problem.result().point_solution()
            # In the bi-objective case, we already know that the regions on the right-hand-side of the selected
            # search problem cannot have a nondominated point as we generete points in non-increasing values in the 
            # x-axis objective
            if self._momilp_model.biobjective():
                search_problems[:] = search_problems[:selected_search_problem_index + 1]
            # solve the slice problem
            slice_problem_result = self._solve_slice_problem(
                iteration_index, selected_region, selected_point_solution)
            frontier = slice_problem_result.frontier_solution().frontier()
            shifted_frontier = SearchUtilities.shift_frontier_in_two_dimension(frontier, alpha=rel_coverage_gap)
            shifted_ideal_point = SearchUtilities.shift_point(
                slice_problem_result.ideal_point(), alpha=rel_coverage_gap)       
            # update the search space
            self._update_lower_bounds(shifted_ideal_point, search_problems, selected_search_problem_index)
            # update the state
            self._update_state(selected_point_solution, frontier, iteration_index)
            # partition the selected region and create the new search problems
            child_search_problems = self._partition_search_problem(shifted_frontier, selected_search_problem)
            search_space.delete_search_problem(selected_search_problem_index)
            for i, p in enumerate(child_search_problems):
                search_space.add_search_problem(p, index = selected_search_problem_index + i)
            # couple search problems
            state.search_space().couple_search_problems()
            # log the progress
            logging.info("Iteration '%d' with '%s'" % (iteration_index, selected_point_solution.point()))
            # log the details in the debug mode
            if debug_mode:
                log = {
                    "selected region": selected_region,
                    "solution": selected_point_solution,
                    "frontier": frontier
                }
                ConeBasedSearchAlgorithm._log_iteration_status(**log)
            # create and add the iteration to the state
            iteration_end_time = time()
            iteration_time_in_seconds = iteration_end_time - iteration_start_time
            self._add_iteration(iteration_index, iteration_time_in_seconds, selected_point_solution)
            # update the iteration index
            iteration_index += 1
        return self._state
