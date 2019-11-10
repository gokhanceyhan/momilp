"""Implements tests for momilp solver"""

from hamcrest import assert_that, is_
import gurobipy
import os
from src.momilp.executor import Executor
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SliceProblem
from unittest import TestCase

class ConeBasedSearchAlgorithmTest(TestCase):

    """Implements tests for the cone-based search algorithm"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 

    def test_three_obj_binary_linear_programming_problem(self):
        """Tests the algorithm on a three-objective binary linear program"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name)
        model.solve()
        assert_that(model.num_obj(), is_(3))
        executor = Executor(model)
        executor.execute()
        y = model.int_vars()
        y_bar = [1] * len(y)
        sp = SliceProblem(model)
        sp.solve(y_bar)

    def test_three_obj_linear_programming_problem(self):
        """Tests the algorithm on a three-objective linear program"""
        pass
        