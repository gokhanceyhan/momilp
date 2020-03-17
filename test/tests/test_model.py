"""Implements tests for model"""

from hamcrest import assert_that, has_key, is_
import gurobipy
import os
from src.momilp.model import GurobiMomilpModel
from unittest import main, TestCase

class GurobiMomilpModelTest(TestCase):

    """Implements tests for the Gurobi Momilp Model"""

    # MOMILP_TO_DO: Add an example model with at least one objective having negative values at minimum
    # MOMILP_TO_DO: Add an example model with at least one objective having unbounded minimum
    # MOMILP_TO_DO: Add an example model with at least one objective having unbounded maximum
    # MOMILP_TO_DO: Add an exmaple model which is infeasible

    # MOMILP_TO_DO: Test constraint deletion

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_create_three_obj_binary_linear_programming_problem(self):
        """Tests the creation of a three-objective binary linear program"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = GurobiMomilpModel(file_name, scale=False)
        self.assert_that(model.num_obj(), is_(3))
        self.assert_that(model.X(), has_key("Budget"))
        self.assert_that(model.X(), has_key("Set0"))
        self.assert_that(model.X(), has_key("Set1"))
        self.assert_that(model.X(), has_key("Set2"))
        self.assert_that(model.Z(), has_key("Set0"))
        self.assert_that(model.Z(), has_key("Set1"))
        self.assert_that(model.Z(), has_key("Set2"))
        for objective_name, objective_range in model.objective_name_2_range().items():
            if objective_name == "Set0":
                self.assert_that(objective_range.min_point_solution().point().values(), is_([0.0, 0.0, 0.0]))
                self.assert_that(objective_range.max_point_solution().point().values(), is_([10.0, 7.0, 6.0]))
                expected_y_bar = [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.assert_that(objective_range.max_point_solution().y_bar(), is_(expected_y_bar))
            if objective_name == "Set1":
                self.assert_that(objective_range.min_point_solution().point().values(), is_([0.0, 0.0, 0.0]))
                self.assert_that(objective_range.max_point_solution().point().values(), is_([7.0, 10.0, 7.0]))
                expected_y_bar = [
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.assert_that(objective_range.max_point_solution().y_bar(), is_(expected_y_bar))
            if objective_name == "Set2":
                self.assert_that(objective_range.min_point_solution().point().values(), is_([0.0, 0.0, 0.0]))
                self.assert_that(objective_range.max_point_solution().point().values(), is_([7.0, 8.0, 9.0]))
                expected_y_bar = [
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                self.assert_that(objective_range.max_point_solution().y_bar(), is_(expected_y_bar))
        for objective_name in model.objective_index_2_name().values():
            objective_scaler = model.objective_scaler(objective_name)
            if objective_name == "Set0":
                self.assert_that(objective_scaler(10), is_(10))
            if objective_name == "Set1":
                self.assert_that(objective_scaler(10), is_(10))
            if objective_name == "Set2":
                self.assert_that(objective_scaler(10), is_(10))


if __name__ == '__main__':
    main()
