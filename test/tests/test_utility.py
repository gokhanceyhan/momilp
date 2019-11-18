"""Implements tests for momilp solver utilities"""

from gurobipy import GRB, Model
from hamcrest import assert_that, has_length, is_
import os
from src.momilp.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, LowerBoundInTwoDimension, \
    PointInTwoDimension, RayInTwoDimension
from src.momilp.model import GurobiMomilpModel
from src.momilp.utility import ConstraintGenerator
from unittest import TestCase

class ConstraintGeneratorTest(TestCase):

    """Implements constraint generator tests"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data")
        self.assert_that = assert_that

    def test_create_constraints_for_cone_in_positive_quadrant(self):
        """Tests the constraint creation for a cone in positive quadrant"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        momilp_model = GurobiMomilpModel(file_name=file_name)
        Z = momilp_model.Z()
        x_var = Z["Set0"]
        y_var = Z["Set1"]
        origin = PointInTwoDimension([0, 0])
        cone = ConvexConeInPositiveQuadrant([RayInTwoDimension(37, origin), RayInTwoDimension(45, origin)])
        constraints = ConstraintGenerator.create_constraints_for_cone_in_positive_quadrant(
            momilp_model, cone, x_var, y_var, name="cone")
        momilp_model.update_model()
        self.assert_that(constraints, has_length(2))
        constraint = constraints[0]
        self.assert_that(round(constraint.rhs, 2), is_(0.0))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, x_var), 2), is_(1.00))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, y_var), 2), is_(-1.00))
        self.assert_that(constraint.sense, is_(GRB.GREATER_EQUAL))
        constraint = constraints[1]
        self.assert_that(round(constraint.rhs, 2), is_(0.0))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, x_var), 2), is_(0.75))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, y_var), 2), is_(-1.00))
        self.assert_that(constraint.sense, is_(GRB.LESS_EQUAL))
        # test the convex cone 'R_{>=0}'
        cone = ConvexConeInPositiveQuadrant([RayInTwoDimension(0, origin), RayInTwoDimension(90, origin)])
        constraints = ConstraintGenerator.create_constraints_for_cone_in_positive_quadrant(
            momilp_model, cone, x_var, y_var, name="positive_quad")
        momilp_model.update_model()
        self.assert_that(constraints, has_length(2))

    def test_create_constraint_for_edge_in_two_dimension(self):
        """Tests the constraint creation for an edge in two dimension"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        momilp_model = GurobiMomilpModel(file_name=file_name)
        Z = momilp_model.Z()
        x_var = Z["Set0"]
        y_var = Z["Set1"]
        edge = EdgeInTwoDimension(PointInTwoDimension([6, 6]), PointInTwoDimension([7, 5.275]))
        constraint = ConstraintGenerator.create_constraint_for_edge_in_two_dimension(momilp_model, edge, x_var, y_var)
        momilp_model.update_model()
        self.assert_that(constraint)
        self.assert_that(round(constraint.rhs, 2), is_(10.35))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, x_var), 2), is_(0.72))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, y_var), 2), is_(1.00))
        # test an edge with infinite slope
        edge = EdgeInTwoDimension(PointInTwoDimension([6, 6]), PointInTwoDimension([6, 3]))
        constraint = ConstraintGenerator.create_constraint_for_edge_in_two_dimension(momilp_model, edge, x_var, y_var)
        momilp_model.update_model()
        self.assert_that(constraint)
        self.assert_that(round(constraint.rhs, 2), is_(6.00))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, x_var), 2), is_(1.00))
        self.assert_that(round(momilp_model.model().getCoeff(constraint, y_var), 2), is_(0.00))
        

    def test_create_constraints_for_lower_bound(self):
        """Tests the constraint creation for a lower bound vector in two dimension"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        momilp_model = GurobiMomilpModel(file_name=file_name)
        Z = momilp_model.Z()
        x_var = Z["Set0"]
        y_var = Z["Set1"]
        lb = LowerBoundInTwoDimension([6.5, 6])
        constraints = ConstraintGenerator.create_constraints_for_lower_bound_in_two_dimension(
            momilp_model, lb, x_var, y_var)
        momilp_model.update_model()
        self.assert_that(constraints, has_length(2))
        self.assert_that(round(constraints[0].rhs, 2), is_(6.50))
        self.assert_that(round(constraints[1].rhs, 2), is_(6.00))
