"""Implements state of the algorithm and its elements"""

from src.common.elements import EdgeSolution, FrontierInTwoDimension
from src.momilp.dominance import DominanceRules
from src.momilp.utilities import TypeConversionUtilities


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


class SolutionState:

    """Implements solution state of the algorithm"""

    def __init__(
            self, nondominated_edges=None, nondominated_points=None, weakly_nondominated_edges=None, 
            weakly_nondominated_points=None):
        self._nondominated_edges = nondominated_edges or []
        self._nondominated_points = nondominated_points or []
        self._weakly_nondominated_edges = weakly_nondominated_edges or []
        self._weakly_nondominated_points = weakly_nondominated_points or []

    def add_nondominated_edge(self, edge_solution):
        """Adds the edge solution to the nondominated edges"""
        self._nondominated_edges.append(edge_solution)

    def add_nondominated_point(self, point_solution):
        """Adds the point solution to the nondominated points"""
        self._nondominated_points.append(point_solution)

    def add_weakly_nondominated_edge(self, edge_solution):
        """Adds the edge solution to the weakly nondominated edges"""
        self._weakly_nondominated_edges.append(edge_solution)

    def filter_dominated_points_and_edges(
            self, dominance_filter, frontier, projected_space_criterion_index_2_criterion_index):
        """Filters the dominated points and edges from the weakly dominated points and edges"""
        assert isinstance(frontier, FrontierInTwoDimension)
        dimensions = sorted(projected_space_criterion_index_2_criterion_index.values())
        dominance_filter.set_dominated_space(frontier)
        for point_solution in self._weakly_nondominated_points:
            point = point_solution.point()
            point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(dimensions, point)
            if DominanceRules.PointToFrontier.dominated(point_in_two_dimension, frontier):
                self._weakly_nondominated_points.remove(point_solution)
        for edge_solution in self._weakly_nondominated_edges:
            self._weakly_nondominated_edges.remove(edge_solution)
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
            filtered_edges_in_two_dimension = dominance_filter.filter_edge(edge_in_two_dimension)
            filtered_edges = [
                TypeConversionUtilities.edge_in_two_dimension_to_edge(
                    unprojected_dim_2_value, e, projected_space_criterion_index_2_criterion_index)  for e in 
                filtered_edges_in_two_dimension]
            nondominated_edges = [EdgeSolution(edge, edge_solution.y_bar()) for edge in filtered_edges]
            self._weakly_nondominated_edges.extend(nondominated_edges)

    def move_weakly_nondominated_to_nondominated(self):
        """Moves all of the weakly nondominated points or edges to nondominated points or edges"""
        self._nondominated_edges.extend(self._weakly_nondominated_edges)
        self._weakly_nondominated_edges = []
        self._nondominated_points.extend(self._weakly_nondominated_points)
        self._weakly_nondominated_points = []

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
