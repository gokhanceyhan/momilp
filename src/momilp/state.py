"""Implements state of the algorithm and its elements"""

import math

from src.common.elements import Edge, EdgeSolution, FrontierInTwoDimension
from src.momilp.dominance import DominanceRules
from src.momilp.utilities import EdgeComparisonUtilities, TypeConversionUtilities


class Iteration:

    """Implements iteration state of the algorithm"""

    def __init__(self, index=0, selected_point_solution=None, statistics=None):
        self._index = index
        self._selected_point_solution = selected_point_solution
        self._statistics = statistics

    def index(self):
        """Returns the index of the iteration of the algorithm"""
        return self._index

    def selected_point_solution(self):
        """Returns the selected point solution"""
        return self._selected_point_solution

    def statistics(self):
        """Returns the statistics with respect to the iteration"""
        return self._statistics


class IterationStatistics:

    """Implements iteration statistics"""

    def __init__(
            self, average_num_tabu_constraints=0, elapsed_time_in_seconds=0, 
            elapsed_time_in_seconds_for_search_problem=0, elapsed_time_in_seconds_for_slice_problem=0, 
            num_milp_solved=0, num_search_problems=0):
        self._average_num_tabu_constraints = average_num_tabu_constraints
        self._elapsed_time_in_seconds = elapsed_time_in_seconds
        self._elapsed_time_in_seconds_for_search_problem = elapsed_time_in_seconds_for_search_problem
        self._elapsed_time_in_seconds_for_slice_problem = elapsed_time_in_seconds_for_slice_problem
        self._num_milp_solved = num_milp_solved
        self._num_search_problems = num_search_problems

    def __str__(self):
        return str({k[1:]: v for k, v in self.__dict__.items()})

    def average_num_tabu_constraints(self):
        """Returns the average number of tabu-constraints in the search problems of the iteration"""
        return self._average_num_tabu_constraints

    def elapsed_time_in_seconds(self):
        """Returns the elapsed time in seconds"""
        return self._elapsed_time_in_seconds

    def elapsed_time_in_seconds_for_search_problem(self):
        """Returns the elapsed time in seconds to solve the search problems"""
        return self._elapsed_time_in_seconds_for_search_problem

    def elapsed_time_in_seconds_for_slice_problem(self):
        """Returns the elapsed time in seconds to solve the slice problems"""
        return self._elapsed_time_in_seconds_for_slice_problem

    def num_milp_solved(self):
        """Returns the number of milp models solved in the iteration"""
        return self._num_milp_solved

    def num_search_problems(self):
        """Returns the number of search problems in the iteration"""
        return self._num_search_problems


class SolutionState:

    """Implements solution state of the algorithm"""

    def __init__(
            self, efficient_integer_vectors=None, nondominated_edges=None, nondominated_points=None, 
            weakly_nondominated_edges=None, weakly_nondominated_points=None):
        # we store the efficient integer vectors as a set of tuples as the edges in the frontier related to an 
        # efficient integer vector are stored individually. 
        self._efficient_integer_vectors = efficient_integer_vectors or set()
        self._nondominated_edges = nondominated_edges or []
        self._nondominated_points = nondominated_points or []
        self._weakly_nondominated_edges = weakly_nondominated_edges or []
        self._weakly_nondominated_points = weakly_nondominated_points or []

    def add_efficient_integer_vector(self, y_bar):
        """Adds the integer vector 'y_bar' to the list of efficient integer vectors
        
        NOTE: 'y_bar' is converted to tuple"""
        self._efficient_integer_vectors.add(tuple(y_bar))

    def add_nondominated_edge(self, edge_solution):
        """Adds the edge solution to the nondominated edges"""
        self._nondominated_edges.append(edge_solution)

    def add_nondominated_point(self, point_solution):
        """Adds the point solution to the nondominated points"""
        self._nondominated_points.append(point_solution)

    def add_weakly_nondominated_edge(
            self, edge_solution, check_continuity=True, constant_value_index=0, to_right=False):
        """Adds the edge solution to the weakly nondominated edges"""
        if not check_continuity or not self._weakly_nondominated_edges:
            self._weakly_nondominated_edges.append(edge_solution)
            return
        previous_edge_solution = self._weakly_nondominated_edges[-1]
        # the edge solution must have the same integer solution vector
        if previous_edge_solution.y_bar() != edge_solution.y_bar():
            self._weakly_nondominated_edges.append(edge_solution)
            return
        base_edge, compared_edge = (previous_edge_solution.edge(), edge_solution.edge()) if to_right else \
            (edge_solution.edge(), previous_edge_solution.edge())
        extended, extended_edge = EdgeComparisonUtilities.extend_edge(base_edge, compared_edge, 
            constant_value_index=constant_value_index)
        if not extended:
            self._weakly_nondominated_edges.append(edge_solution)
            return
        edge_solution_ = EdgeSolution(extended_edge, previous_edge_solution.y_bar())
        self._weakly_nondominated_edges[-1] = edge_solution_

    def efficient_integer_vectors(self):
        """Returns the efficient integer vectors"""
        return self._efficient_integer_vectors

    def filter_dominated_points_and_edges(
            self, dominance_filter, frontier, projected_space_criterion_index_2_criterion_index):
        """Filters the dominated points and edges from the weakly dominated points and edges"""
        assert isinstance(frontier, FrontierInTwoDimension)
        dimensions = sorted(projected_space_criterion_index_2_criterion_index.values())
        dominated_space_set_in_model = False
        for point_solution in self._weakly_nondominated_points:
            point = point_solution.point()
            point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(dimensions, point)
            if DominanceRules.PointToFrontier.dominated(point_in_two_dimension, frontier):
                self._weakly_nondominated_points.remove(point_solution)
        filtered_weakly_nondominated_edges = []
        for edge_solution in self._weakly_nondominated_edges:
            edge = edge_solution.edge()
            num_dim = len(edge.start_point().values())
            assert all(
                [edge.start_point().values()[i] == edge.end_point().values()[i] for i in range(num_dim) if i not in 
                 dimensions]), "the points of the edge must have the same values in the dimensions other than the %s" \
                "dimensions" % dimensions
            edge_in_two_dimension = TypeConversionUtilities.edge_to_edge_in_two_dimension(dimensions, edge)
            unprojected_dim_2_value = {
                index: value for index, value in enumerate(edge.start_point().values()) if index not in dimensions}
            if DominanceRules.EdgeToFrontier.dominated(edge_in_two_dimension, frontier):
                continue
            if not dominated_space_set_in_model:
                dominance_filter.set_dominated_space(frontier)
                dominated_space_set_in_model = True
            filtered_edges_in_two_dimension = dominance_filter.filter_edge(edge_in_two_dimension)
            filtered_edges = [
                TypeConversionUtilities.edge_in_two_dimension_to_edge(
                    unprojected_dim_2_value, e, projected_space_criterion_index_2_criterion_index)  for e in 
                filtered_edges_in_two_dimension]
            nondominated_edges = [EdgeSolution(edge, edge_solution.y_bar()) for edge in filtered_edges]
            # add the generated edges in reverse order to maintain the order from right-to-left in x_obj values
            filtered_weakly_nondominated_edges.extend(nondominated_edges[::-1])
        self._weakly_nondominated_edges = filtered_weakly_nondominated_edges

    def filter_weakly_nondominated_edges(self, criterion_index=0, criterion_value=0):
        """Checks each edge in the weakly nondominated set, and moves them to the nondominated set if they 
        have higher value in the specified criterion assuming that the condition proves the nondominance"""
        nondominated_edge_solutions = []
        for edge_solution in self._weakly_nondominated_edges:
            edge = edge_solution.edge()
            if edge.start_point().values()[criterion_index] <= criterion_value:
                continue
            nondominated_edge_solutions.append(edge_solution)
        efficient_integer_vectors = [p.y_bar() for p in nondominated_edge_solutions]
        self._weakly_nondominated_edges = [
            e for e in self._weakly_nondominated_edges if e not in nondominated_edge_solutions]
        self._nondominated_edges.extend(nondominated_edge_solutions)
        tuples = [tuple(e) for e in efficient_integer_vectors]
        self._efficient_integer_vectors.update(tuples)

    def filter_weakly_nondominated_points(self, criterion_index=0, criterion_value=0):
        """Checks each point in the weakly nondominated set, and moves them to the nondominated set if they 
        have higher value in the specified criterion assuming that the condition proves the nondominance"""
        nondominated_point_solutions = []
        for point_solution in self._weakly_nondominated_points:
            point = point_solution.point()
            if point.values()[criterion_index] <= criterion_value:
                continue
            nondominated_point_solutions.append(point_solution)
        efficient_integer_vectors = [p.y_bar() for p in nondominated_point_solutions]
        self._weakly_nondominated_points = [
            p for p in self._weakly_nondominated_points if p not in nondominated_point_solutions]
        self._nondominated_points.extend(nondominated_point_solutions)
        tuples = [tuple(e) for e in efficient_integer_vectors]
        self._efficient_integer_vectors.update(tuples)

    def move_weakly_nondominated_edges_to_nondominated_edges(
            self, check_continuity=True, constant_value_index=0, to_right=False):
        """Moves the weakly nondominated edges to the set of nondominated edges"""
        if not self._weakly_nondominated_edges:
            return
        if not check_continuity or not self._nondominated_edges:
            self._nondominated_edges.extend(self._weakly_nondominated_edges)
            return
        previous_edge_solution = self._nondominated_edges[-1]
        edge_solution = self._weakly_nondominated_edges[0]
        # the edge solution must have the same integer solution vector
        if previous_edge_solution.y_bar() != edge_solution.y_bar():
            self._nondominated_edges.extend(self._weakly_nondominated_edges)
            return
        base_edge, compared_edge = (previous_edge_solution.edge(), edge_solution.edge()) if to_right else \
            (edge_solution.edge(), previous_edge_solution.edge())
        extended, extended_edge = EdgeComparisonUtilities.extend_edge(base_edge, compared_edge, 
            constant_value_index=constant_value_index)
        if not extended:
            self._nondominated_edges.extend(self._weakly_nondominated_edges)
            return
        edge_solution_ = EdgeSolution(extended_edge, previous_edge_solution.y_bar())
        self._nondominated_edges[-1] = edge_solution_
        if len(self._weakly_nondominated_edges) > 1:
            self._nondominated_edges.extend(self._weakly_nondominated_edges[1:])

    def move_weakly_nondominated_to_nondominated(self, constant_value_index=0):
        """Moves all of the weakly nondominated points or edges to nondominated points or edges"""
        self.move_weakly_nondominated_edges_to_nondominated_edges(constant_value_index=constant_value_index)
        efficient_integer_vectors_of_edge_solutions = [e.y_bar() for e in self._weakly_nondominated_edges]
        self._weakly_nondominated_edges = []
        self._nondominated_points.extend(self._weakly_nondominated_points)
        efficient_integer_vectors_of_point_solutions = [p.y_bar() for p in self._weakly_nondominated_points]
        self._weakly_nondominated_points = []
        efficient_integer_vectors = \
            efficient_integer_vectors_of_edge_solutions + efficient_integer_vectors_of_point_solutions
        tuples = [tuple(e) for e in efficient_integer_vectors]
        self._efficient_integer_vectors.update(tuples)

    def nondominated_edges(self):
        """Returns the nondominated edges"""
        return self._nondominated_edges

    def nondominated_points(self):
        """Returns the nondominated points"""
        return self._nondominated_points


class State:

    """Implements state of the algorithm"""

    def __init__(self, search_space, slice_problem, iterations=None, solution_state=None):
        self._iterations = iterations or []
        self._search_space = search_space
        self._slice_problem = slice_problem
        self._solution_state = solution_state

    def iterations(self):
        """Returns the iterations"""
        return self._iterations

    def search_space(self):
        """Returns the search space"""
        return self._search_space

    def slice_problem(self):
        """Returns the slice problem"""
        return self._slice_problem

    def solution_state(self):
        """Returns the solution state"""
        return self._solution_state
