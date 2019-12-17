"""Implements algorithm to solve the momilp"""

import abc
import copy
from enum import Enum
import operator
import os
from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, EdgeSolution, RayInTwoDimension, \
    FrontierInTwoDimension, FrontierSolution, OptimizationStatus, PointInTwoDimension, PointSolution, \
    SearchRegionInTwoDimension, SliceProblemResult
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
    def _create_cone_based_search_algorithm(model):
        """Creates and returns the cone-based search algorithm"""
        return ConeBasedSearchAlgorithm(model)

    @staticmethod
    def create(model, algorithm_type=AlgorithmType.CONE_BASED_SEARCH):
        """Creates algorithm"""
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
        if algorithm_type == AlgorithmType.CONE_BASED_SEARCH:
            return AlgorithmFactory._create_cone_based_search_algorithm(model)


class ConeBasedSearchAlgorithm(AbstractAlgorithm):

    """Implements the cone-based search algorithm"""

    def __init__(self, configuration, model, working_dir, discrete_objective_indices=None):
        self._configuration = configuration
        self._discreate_objective_indices = discrete_objective_indices or []
        self._errors = []
        self._model = model
        self._objective_index_2_priority = {}
        self._primary_objective_index = None
        self._state = None
        self._working_dir = working_dir
        self._initialize()

    def _create_momilp_model(self, **kwargs):
        """Wraps the model as a momilp model and returns it"""
        return GurobiMomilpModel(
            gurobi_model=self._model.copy(), discrete_objective_indices=self._discreate_objective_indices)

    def _create_positive_quadrant_convex_cone(self):
        """Returns a convex cone corresponding to the positive quadrant"""
        return ConvexConeInPositiveQuadrant(
            [RayInTwoDimension(90, PointInTwoDimension([0, 0])), RayInTwoDimension(0, PointInTwoDimension([0, 0]))])

    def _initialize(self):
        """Initializes the algorithm"""
        # create the initial search problem
        cone = self._create_positive_quadrant_convex_cone()
        region = SearchRegionInTwoDimension(cone)
        try:
            search_problem = SearchProblem(self._create_momilp_model())
        except BaseException as e:
            raise RuntimeError(
                "failed to create the search problem in the initialization of the cone-based search algorithm") from e
        search_problem.update_model(region=region)
        # set the primary objective index and objective priorities
        momilp_model = search_problem.momilp_model()
        self._primary_objective_index = momilp_model.primary_objective_index()
        self._objective_index_2_priority = momilp_model.objective_index_2_priority()
        # create the initial search space
        sorting_criteria_index = [
            i for i in self._objective_index_2_priority.keys() if i != self._primary_objective_index][0]
        search_space = SearchSpace(self._primary_objective_index, sorting_criteria_index)
        search_space.add_search_problem(search_problem)
        # create a slice problem
        try:
            slice_problem = SliceProblem(self._create_momilp_model())
        except BaseException as e:
            raise RuntimeError(
                "failed to create the slice problem in the initialization of the cone-based search algorithm") from e
        # create the initial solution state
        solution_state = SolutionState()
        self._state = State(search_space, slice_problem, solution_state=solution_state)

    def run(self):
        state = self._state
        search_space = state.search_space()
        search_problems = search_space.search_problems()
        slice_problem = state.slice_problem()
        iteration_index = 0
        while search_problems:
            # search all of the regions and remove the infeasible ones
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
            search_problems[:] = [
                p for p in search_problems if p.result().status() in 
                [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]]
            if not search_problems:
                state.solution_state().move_weakly_nondominated_to_nondominated()
                break
            selected_search_problem_and_index = None
            for search_problem, index in enumerate(search_problems):
                if not selected_search_problem_and_index:
                    selected_search_problem_and_index = (search_problem, index)
                else:
                    selected_search_problem = operator.itemgetter(selected_search_problem_and_index, 0)
                    selected_point = selected_search_problem.result().point_solution().point()
                    candiate_point = search_problem.result().point_solution().point()
                    if PointComparisonUtilities.compare_to(
                            selected_point, candiate_point, self._objective_index_2_priority) < 0:
                        selected_search_problem_and_index = (search_problem, index)
            
            # select the next efficient integer vector, solve the slice problem and update the search space
            selected_search_problem = operator.itemgetter(selected_search_problem_and_index, 0)
            selected_search_problem_index = operator.itemgetter(selected_search_problem_and_index, 1)
            selected_point_solution = selected_search_problem.result().point_solution()
            slice_problem.update_model(
                selected_point_solution.y_bar(), 
                primary_objective_value=selected_point_solution.point().values()[self._primary_objective_index], 
                region=selected_search_problem.region())
            try:
                slice_problem_result = slice_problem.solve()
            except BaseException as e:
                raise RuntimeError(
                    "failed to solve the slice problem for integer vector '%s' in iteration '%s'") % (
                        selected_point_solution.y_bar(), iteration_index) from e
            search_space.update_lower_bounds(
                slice_problem_result.frontier_solution().ideal_point(), selected_search_problem_index)

            # update the state
            if iteration_index > 0:
                previous_selected_point_solution = state.iterations()[iteration_index - 1].selected_point_solution()
                if selected_point_solution.point().values()[self._primary_objective_index] < \
                        previous_selected_point_solution.point().values()[self._primary_objective_index]:
                    state.solution_state().move_weakly_nondominated_to_nondominated()
                else:
                    state.solution_state().filter_dominated_points_and_edges(slice_problem_result.frontier_solution())
            frontier = slice_problem_result.frontier_solution().frontier()
            y_bar = slice_problem_result.frontier_solution().y_bar()
            if frontier.singleton():
                state.solution_state().add_nondominated_point(selected_point_solution)
            else:
                edges = frontier.edges()
                for edge in edges:
                    state.solution_state().add_weakly_nondominated_edge(EdgeSolution(edge, y_bar))
            
            # partition the selected search region
            selected_region = selected_search_problem.region()
            child_search_regions = SearchUtilities.partition_search_region_in_two_dimension(frontier, selected_region)
            for region in child_search_regions:
                search_problem = selected_search_problem.copy()
                search_problem.update_model(region=region, tabu_y_bars=[y_bar], keep_previous_tabu_constraints=True)
                search_space.add_search_problem(search_problem)
            search_space.delete_search_problem(selected_search_problem_index)
            