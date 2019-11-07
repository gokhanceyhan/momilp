"""Implements state of the algorithm and its elements"""

class Iteration:

    """Implements iteration state of the algorithm"""

    def __init__(self, index=0, search_problem_index=0, statistics=None):
        self._index = index
        self._search_problem_index = search_problem_index
        self._statistics = statistics


class SolutionState:

    """Implements solution state of the algorithm"""

    def __init__(self, efficient_set, nondominated_set, weakly_efficient_set, weakly_nondominated_set):
        self._efficient_set = efficient_set
        self._nondomianted_set = nondominated_set
        self._weakly_efficient_set = weakly_efficient_set
        self._weakly_nondominated_set = weakly_nondominated_set


class State:

    """Implements state of the algorithm"""

    def __init__(self, search_space, slice_problem, iterations=None, solution_state=None):
        self._iterations = iterations or []
        self._search_space = search_space
        self._slice_problem = slice_problem
        self._solution_state = solution_state