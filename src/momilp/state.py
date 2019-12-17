"""Implements state of the algorithm and its elements"""


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

    def filter_dominated_points_and_edges(self, frontier):
        """Filters the dominated points and edges from the weakly dominated points and edges"""
        pass

    def move_weakly_nondominated_to_nondominated(self):
        """Moves all of the weakly nondominated points or edges to nondominated points or edges"""
        self._nondominated_edges.extend(self._weakly_nondominated_edges)
        self._weakly_nondominated_edges = []
        self._nondominated_points.extend(self._weakly_nondominated_points)
        self._weakly_nondominated_points = []        


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
