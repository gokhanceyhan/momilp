"""Implements algorithm to solve the momilp"""

import abc
import copy
from enum import Enum
from gurobipy import read
import operator
import os
from src.common.elements import ConvexConeInPositiveQuadrant, Edge, EdgeInTwoDimension, EdgeSolution, \
    RayInTwoDimension, FrontierInTwoDimension, FrontierSolution, OptimizationStatus, Point, PointInTwoDimension, \
    PointSolution, SearchRegionInTwoDimension, SliceProblemResult
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SearchProblem, SearchSpace, SliceProblem
from src.momilp.state import Iteration, SolutionState, State
from src.momilp.utilities import PointComparisonUtilities, SearchUtilities


class AbstractAlgorithm(metaclass=abc.ABCMeta):

    """Implements abstract class for the algorithm"""
    
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
            model_file, working_dir, discrete_objective_indices=None, explore_decision_space=False):
        """Creates and returns the cone-based search algorithm"""
        return ConeBasedSearchAlgorithm(
            model_file, working_dir, discrete_objective_indices=discrete_objective_indices, 
            explore_decision_space=explore_decision_space)

    @staticmethod
    def create(
            model_file, working_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, discrete_objective_indices=None, 
            explore_decision_space=False):
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
                model_file, working_dir, discrete_objective_indices=discrete_objective_indices, 
                explore_decision_space=explore_decision_space)


class ConeBasedSearchAlgorithm(AbstractAlgorithm):

    """Implements the cone-based search algorithm"""

    _LOWER_BOUND_DELTA = 1e-3

    def __init__(self, model_file, working_dir, discrete_objective_indices=None, explore_decision_space=False):
        self._discrete_objective_indices = discrete_objective_indices or []
        self._errors = []
        self._explore_decision_space = explore_decision_space
        self._model_file = model_file
        self._momilp_model = None
        self._objective_index_2_priority = {}
        self._primary_objective_index = None
        self._projected_space_criterion_index_2_criterion_index = {}
        self._state = None
        self._x_obj_name = None
        self._y_obj_name = None
        self._working_dir = working_dir
        self._initialize()

    def _convert_edge_in_projected_space_to_edge_original_space(self, edge):
        """Returns an edge in the original search space from the edge in the projected search space"""
        start_point = {
            self._projected_space_criterion_index_2_criterion_index[index]: value for index, value in 
            enumerate(edge.left_point().values())}
        start_point[self._primary_objective_index] = edge.z3()
        end_point = {
            self._projected_space_criterion_index_2_criterion_index[index]: value for index, value in 
            enumerate(edge.right_point().values())}
        end_point[self._primary_objective_index] = edge.z3()
        return Edge(Point(list(start_point.values())), Point(end_point.values()))

    def _create_momilp_model(self):
        """Creates and returns a momilp model"""
        return GurobiMomilpModel(
            file_name=self._model_file, discrete_objective_indices=self._discrete_objective_indices)

    def _create_positive_quadrant_convex_cone(self):
        """Returns a convex cone corresponding to the positive quadrant"""
        return ConvexConeInPositiveQuadrant(
            [RayInTwoDimension(90, PointInTwoDimension([0, 0])), RayInTwoDimension(0, PointInTwoDimension([0, 0]))])

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
            search_problem = SearchProblem(self._create_momilp_model())
        except BaseException as e:
            raise RuntimeError(
                "failed to create the search problem in the initialization of the cone-based search algorithm") from e
        search_problem.update_model(region=region)
        # create a slice problem
        try:
            slice_problem = SliceProblem(
                self._create_momilp_model(), self._projected_space_criterion_index_2_criterion_index)
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
    
    @staticmethod
    def _is_problem_feasible(problem):
        """Returns True if the problem is feasible, False othwerwise"""
        return problem.result().status() in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

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

    def _solve_search_problems(self, iteration_index, search_problems):
        """Solves and returns the search problems"""
        for search_problem in search_problems:
            if search_problem.result():
                continue
            try:
                search_problem.solve()
            except BaseException as e:
                self._errors.append(
                    "the problem '%s' failed with error '%s' in iteration '%s'" % (
                        id(search_problem), e, iteration_index))
                search_problem.momilp_model().write(os.path.join(self._working_dir, id(search_problem) + ".lp"))     
        search_problems[:] = [p for p in search_problems if ConeBasedSearchAlgorithm._is_problem_feasible(p)]
        return search_problems

    def _solve_slice_problem(self, selected_point_solution, region, iteration_index):
        """Solves the slice problem and returns the result"""
        slice_problem = self._state.slice_problem()
        slice_problem.update_model(
            selected_point_solution.y_bar(), 
            primary_objective_value=selected_point_solution.point().values()[self._primary_objective_index], 
            region=region)
        try:
            slice_problem.momilp_model().write("./logs/slice.lp")
            return slice_problem.solve()
        except BaseException as e:
            raise RuntimeError(
                "failed to solve the slice problem for integer vector '%s' in iteration '%s'" % (
                    selected_point_solution.y_bar(), iteration_index)) from e

    def _update_state(self, selected_point_solution, frontier, iteration_index):
        """Updates the state"""
        state = self._state
        if iteration_index > 0:
            previous_selected_point_solution = state.iterations()[iteration_index - 1].selected_point_solution()
            if selected_point_solution.point().values()[self._primary_objective_index] < \
                    previous_selected_point_solution.point().values()[self._primary_objective_index]:
                state.solution_state().move_weakly_nondominated_to_nondominated()
            else:
                state.solution_state().filter_dominated_points_and_edges(frontier)
        if frontier.singleton():
            state.solution_state().add_nondominated_point(selected_point_solution)
        else:
            edges = frontier.edges()
            for edge in edges:
                solution_edge = self._convert_edge_in_projected_space_to_edge_original_space(edge)
                state.solution_state().add_weakly_nondominated_edge(
                    EdgeSolution(solution_edge, selected_point_solution.y_bar()))

    def momilp_model(self):
        """Returns the momilp model"""
        return self._momilp_model

    def run(self):
        state = self._state
        search_space = state.search_space()
        search_problems = search_space.search_problems()
        lower_bound_delta = ConeBasedSearchAlgorithm._LOWER_BOUND_DELTA if not self._explore_decision_space else 0.0
        iteration_index = 0
        while search_problems:
            # search all of the regions and remove the infeasible ones
            search_problems = self._solve_search_problems(iteration_index, search_problems)

            for index, p in enumerate(search_problems):
                p.momilp_model().write("./logs/iteration_%d_region_%d.lp" %(iteration_index, index))

            if not search_problems:
                state.solution_state().move_weakly_nondominated_to_nondominated()
                break
            # select the next efficient integer vector and solve the slice problem
            selected_search_problem_and_index = self._select_search_problem_and_index(search_problems)
            selected_search_problem = selected_search_problem_and_index[0]
            selected_search_problem_index = selected_search_problem_and_index[1]
            selected_point_solution = selected_search_problem.result().point_solution()
            slice_problem_result = self._solve_slice_problem(
                selected_point_solution, selected_search_problem.region(), iteration_index)
            frontier = slice_problem_result.frontier_solution().frontier()
            y_bar = slice_problem_result.frontier_solution().y_bar()
            # update the search space
            search_space.update_lower_bounds(
                slice_problem_result.ideal_point(), selected_search_problem_index, delta=lower_bound_delta)
            # update the state
            self._update_state(selected_point_solution, frontier, iteration_index)
            # partition the selected search region
            selected_region = selected_search_problem.region()
            child_search_regions = SearchUtilities.partition_search_region_in_two_dimension(
                frontier, selected_region, lower_bound_delta=lower_bound_delta)     
            for child_index, region in enumerate(child_search_regions):
                search_problem = SearchProblem(self._create_momilp_model())
                tabu_y_bars = selected_search_problem.tabu_y_bars()[:]
                tabu_y_bars.append(y_bar)
                search_problem.update_model(region=region, tabu_y_bars=tabu_y_bars)
                search_space.add_search_problem(search_problem, index=selected_search_problem_index + 1 + child_index)
            search_space.delete_search_problem(selected_search_problem_index)
            # update the iteration index
            state.iterations().append(Iteration(iteration_index, selected_point_solution=selected_point_solution))
            iteration_index += 1
        return self._state
