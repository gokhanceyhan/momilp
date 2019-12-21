"""Implements tests for momilp solver algorithm"""

from gurobipy import read
from hamcrest import assert_that, is_
import os
from src.momilp.algorithm import AlgorithmFactory, AlgorithmType
from src.momilp.executor import Executor
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SliceProblem
from unittest import TestCase

class ConeBasedSearchAlgorithmTest(TestCase):

    """Implements tests for the cone-based search algorithm"""

    def setUp(self):
        self._logs_dir = os.environ["MOMILP_LOG_PATH"]
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_three_obj_binary_linear_programming_problem(self):
        """Tests the algorithm on a three-objective binary linear program"""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            discrete_objective_indices=[0, 1, 2])
        algorithm.run()

    def test_three_obj_linear_programming_problem(self):
        """Tests the algorithm on a three-objective linear program"""
        pass
        