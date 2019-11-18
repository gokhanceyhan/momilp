"""Implements tests for momilp solver search"""

from hamcrest import assert_that, is_
import gurobipy
import os
from src.momilp.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, LowerBoundInTwoDimension, \
    PointInTwoDimension, RayInTwoDimension, SearchRegionInTwoDimension
from src.momilp.model import GurobiMomilpModel
from src.momilp.search import SliceProblem
from unittest import TestCase

class SliceProblemTest(TestCase):

    """Implements tests for the slice problem"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_slice_problem_restricted_to_search_region_in_two_dimension(self):
        """Tests slice problem restricted to a search region"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name=file_name)
        slice_problem = SliceProblem(model)
        origin = PointInTwoDimension([0, 0])
        cone = ConvexConeInPositiveQuadrant([RayInTwoDimension(37, origin), RayInTwoDimension(45, origin)])
        edge = EdgeInTwoDimension(PointInTwoDimension([6, 6]), PointInTwoDimension([7, 5.275]))
        lb = LowerBoundInTwoDimension([6.5, 6])
        region = SearchRegionInTwoDimension(cone, edge=edge, lower_bound=lb, x_obj_name="Set0", y_obj_name="Set1")
        y_bar = [1] * 12 + [0] * 8
        slice_problem.solve(y_bar, region=region)

    def test_unrestricted_slice_problem(self):
        """Tests the slice problem without any region constraints"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name=file_name)
        slice_problem = SliceProblem(model)
        y_bar = [1] * 12 + [0] * 8
        slice_problem.solve(y_bar)
        
        