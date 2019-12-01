"""Implements tests for dichotomic search"""

from gurobipy import read
from hamcrest import assert_that, has_length, is_
import os
from src.molp.dichotomic_search.solver import BolpDichotomicSearchWithGurobiSolver, Point
from unittest import TestCase


class BolpDichotomicSearchTest(TestCase):

    """Implements tests for the dichomotic search to solve bolp"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_knapsack_problem(self):
        """Tests the algorithm on a knapsack problem"""
        file_name = os.path.join(self._test_data_dir, "bolp.lp")
        model = read(file_name)
        solver = BolpDichotomicSearchWithGurobiSolver(model)
        solver.solve()
        extreme_supported_nondominated_points = solver.extreme_supported_nondominated_points()
        self.assert_that(extreme_supported_nondominated_points, has_length(6))
        self.assert_that(solver._point_pairs_to_check, has_length(0))
        first_point = extreme_supported_nondominated_points[0]
        self.assert_that(first_point, is_(Point(z1=81, z2=96)))
        last_point = extreme_supported_nondominated_points[-1]
        self.assert_that(last_point, is_(Point(z1=120, z2=17)))