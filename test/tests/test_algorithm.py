"""Implements tests for momilp solver algorithm"""

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
        self.assert_that = assert_that

    def test_three_obj_binary_linear_programming_problem(self):
        """Tests the algorithm on a three-objective binary linear program"""

    def test_three_obj_linear_programming_problem(self):
        """Tests the algorithm on a three-objective linear program"""
        pass
        