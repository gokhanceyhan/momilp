"""Implements tests for momilp solver algorithm"""

from gurobipy import read
from hamcrest import assert_that, empty, has_length, is_
import os
from src.momilp.algorithm import AlgorithmFactory, AlgorithmType
from src.momilp.executor import Executor
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SliceProblem
from src.momilp.utilities import ReportCreator
from unittest import main, TestCase

class ConeBasedSearchAlgorithmTest(TestCase):

    """Implements tests for the cone-based search algorithm"""

    def setUp(self):
        self._instance_name = "instance"
        self._logs_dir = os.environ["MOMILP_LOG_PATH"]
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_three_obj_binary_linear_programming_problem(self):
        """Tests the algorithm on a three-objective binary linear program"""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            discrete_objective_indices=[0, 1, 2], explore_decision_space=False)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        num_points, _ = nondominated_points_df.shape
        self.assert_that(num_points, is_(7))
        self.assert_that(nondominated_edges_df.empty)

    def test_three_obj_binary_linear_programming_problem_with_exploring_decision_space(self):
        """Tests the algorithm on a three-objective binary linear program with exploring the decision space"""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            discrete_objective_indices=[0, 1, 2], explore_decision_space=True, max_num_iterations=50)
        state = algorithm.run()
        solution_state = state.solution_state()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        self.assert_that(solution_state.nondominated_points(), has_length(6))
        self.assert_that(solution_state.nondominated_edges(), empty())
        distinct_points = set([nd_point.point() for nd_point in solution_state.nondominated_points()])
        self.assert_that(list(distinct_points), has_length(2))
        self.assert_that(solution_state.efficient_integer_vectors(), has_length(6))

    def test_three_obj_blp_ex1_problem(self):
        """Tests the algorithm on a small 3-obj blp
        
        NOTE: There is a single binary variable, and one objective has only that variable. The nondominated set
        has two edges."""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp_ex1.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            explore_decision_space=True)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        self.assert_that(nondominated_points_df.empty)
        num_edges = len(state.solution_state().nondominated_edges())
        self.assert_that(num_edges, is_(2))    

    def test_three_obj_blp_ex2_problem(self):
        """Tests the algorithm on a small 3-obj blp
        
        NOTE: There is a single binary variable, and one objective has only that variable. The nondominated set
        has one point and one edge."""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp_ex2.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            explore_decision_space=True)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        num_points = len(state.solution_state().nondominated_points())
        self.assert_that(num_points, is_(1))
        num_edges = len(state.solution_state().nondominated_edges())
        self.assert_that(num_edges, is_(1))

    def test_three_obj_blp_ex3_problem(self):
        """Tests the algorithm on a small 3-obj blp
        
        NOTE: There are two binary variables. The nondominated set has three points."""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp_ex3.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            explore_decision_space=True)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        num_points = len(state.solution_state().nondominated_points())
        self.assert_that(num_points, is_(3))
        self.assert_that(nondominated_edges_df.empty)

    def test_three_obj_blp_ex4_problem(self):
        """Tests the algorithm on a small 3-obj blp
        
        NOTE: There are two binary variables. The nondominated set has 2 edges. This example is nice to examine since 
        some frontiers span more than one region and the same edge is found with different primary criterion values"""
        model_file = os.path.join(self._test_data_dir, "three_obj_blp_ex4.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            explore_decision_space=True)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        num_points = len(state.solution_state().nondominated_points())
        num_edges = len(state.solution_state().nondominated_edges())
        # MOMILP_TO_DO: MOMILP-8: Generation of weakly nondominated but dominated points or edges
        self.assert_that(num_edges, is_(5))
        self.assert_that(num_points, is_(0))

    def test_paper_ex_2_problem(self):
        """Tests the algorithm on Example 2 of the paper"""
        model_file = os.path.join(self._test_data_dir, "paper_ex_1.lp")
        algorithm = AlgorithmFactory.create(
            model_file, self._logs_dir, algorithm_type=AlgorithmType.CONE_BASED_SEARCH, 
            explore_decision_space=True)
        state = algorithm.run()
        report_creator = ReportCreator(algorithm.momilp_model(), state, self._instance_name, self._logs_dir)
        report_creator.create_data_frames()
        nondominated_points_df = report_creator.nondominated_points_df()
        nondominated_edges_df = report_creator.nondominated_edges_df()
        print(nondominated_points_df)
        print(nondominated_edges_df)
        num_points = len(state.solution_state().nondominated_points())
        num_edges = len(state.solution_state().nondominated_edges())
        # MOMILP_TO_DO: Actual number of nondominated edge is 3 but one edge is represented with its two sub-edges. 
        self.assert_that(num_edges, is_(4))
        self.assert_that(num_points, is_(0))

    def test_three_obj_linear_programming_problem(self):
        """Tests the algorithm on a three-objective linear program"""
        pass
        

if __name__ == '__main__':
    main()
