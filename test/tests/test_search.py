"""Implements tests for momilp solver search"""

from hamcrest import assert_that, has_key, is_
from gurobipy import GRB
import os
from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, LowerBoundInTwoDimension, \
    OptimizationStatus, PointInTwoDimension, RayInTwoDimension, SearchRegionInTwoDimension
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SearchProblem, SliceProblem
from unittest import TestCase


class SearchProblemTest(TestCase):

    """Implements tests for the search problem"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data")
        self.assert_that = assert_that

    def test_tabu_constraint_handling(self):
        """Tests handling the tabu-constraints in the problem"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name=file_name)
        search_problem = SearchProblem(model)
        result = search_problem.solve()
        point_solution = result.point_solution()
        # check the unrestricted search problem
        self.assert_that(result.status(), is_(OptimizationStatus.OPTIMAL))
        y_opt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assert_that(point_solution.y_bar(), is_(y_opt))
        # restrict the optimal integer vector
        search_problem.update_model(tabu_y_bars=[y_opt])
        self.assert_that(search_problem.num_tabu_constraints(), is_(1))
        self.assert_that(model.constraint_name_2_constraint(), has_key("tabu_0"))
        result = search_problem.solve()
        point_solution = result.point_solution()
        self.assert_that(result.status(), is_(OptimizationStatus.OPTIMAL))
        y_opt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assert_that(point_solution.y_bar(), is_(y_opt))
        # add one more constraint
        search_problem.update_model(keep_previous_tabu_constraints=True, tabu_y_bars=[y_opt])
        self.assert_that(search_problem.num_tabu_constraints(), is_(2))
        self.assert_that(model.constraint_name_2_constraint(), has_key("tabu_1"))
        result = search_problem.solve()
        point_solution = result.point_solution()
        self.assert_that(result.status(), is_(OptimizationStatus.OPTIMAL))
        y_opt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assert_that(point_solution.y_bar(), is_(y_opt))
        # add a single constraint, and remove the older ones
        search_problem.update_model(tabu_y_bars=[y_opt])
        self.assert_that(search_problem.num_tabu_constraints(), is_(1))
        self.assert_that(model.constraint_name_2_constraint(), has_key("tabu_0"))
        result = search_problem.solve()
        point_solution = result.point_solution()
        self.assert_that(result.status(), is_(OptimizationStatus.OPTIMAL))
        y_opt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assert_that(point_solution.y_bar(), is_(y_opt))
        

class SliceProblemTest(TestCase):

    """Implements tests for the slice problem"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self._slice_prob_obj_index_2_original_obj_index = {0: 1, 1: 2}
        self.assert_that = assert_that

    def test_slice_problem_restricted_to_search_region_in_two_dimension(self):
        """Tests slice problem restricted to a search region"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name=file_name)
        slice_problem = SliceProblem(model, self._slice_prob_obj_index_2_original_obj_index)
        origin = PointInTwoDimension([0, 0])
        cone = ConvexConeInPositiveQuadrant([RayInTwoDimension(37, origin), RayInTwoDimension(45, origin)])
        edge = EdgeInTwoDimension(PointInTwoDimension([6, 6]), PointInTwoDimension([7, 5.275]))
        lb = LowerBoundInTwoDimension([6.5, 6])
        region = SearchRegionInTwoDimension("Set0", "Set1", cone, edge=edge, lower_bound=lb)
        y_bar = [1] * 12 + [0] * 8
        slice_problem.update_model(region=region, y_bar=y_bar)
        slice_problem.solve()
        result = slice_problem.result()
        self.assert_that(result.frontier_solution().frontier().point(), is_(PointInTwoDimension([7, 6])))

    def test_unrestricted_slice_problem(self):
        """Tests the slice problem without any region constraints"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name=file_name)
        slice_problem = SliceProblem(model, self._slice_prob_obj_index_2_original_obj_index)
        y_bar = [1] * 12 + [0] * 8
        slice_problem.update_model(y_bar=y_bar)
        slice_problem.solve()
        result = slice_problem.result()
        self.assert_that(result.frontier_solution().frontier().point(), is_(PointInTwoDimension([7, 6])))
