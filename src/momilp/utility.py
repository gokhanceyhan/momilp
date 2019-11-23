"""Implements utilities for the momilp solver"""

from gurobipy import Constr, GRB
import math
from src.momilp.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, LowerBoundInTwoDimension, \
    Point, PointSolution


class ConstraintGenerationUtilities:

    """Implements constraint generator utility"""

    _EDGE_CONSTRAINT_NAME_PREFIX= "edge"
    _LEFT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "left_extr_ray"
    _LOWER_BOUND_CONSTRAINT_NAME_PREFIX = "lb"
    _RIGHT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "right_extr_ray"

    @staticmethod
    def create_constraints_for_cone_in_positive_quadrant(model, cone, x_var, y_var, name=None):
        """Creates and adds the constraints to the model for the given cone, returns the constraints"""
        assert isinstance(cone, ConvexConeInPositiveQuadrant)
        constraints = []
        name = name or str(id(cone))
        left_extreme_ray = cone.left_extreme_ray()
        x_coeff = math.tan(math.radians(left_extreme_ray.angle_in_degrees())) if \
            left_extreme_ray.angle_in_degrees() < 90 else 1.0
        y_coeff = 1 if left_extreme_ray.angle_in_degrees() < 90 else 0.0
        lhs = x_coeff * x_var - y_coeff * y_var
        rhs = 0.0
        name_ = "_".join([name, ConstraintGenerationUtilities._LEFT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX])
        constraints.append(model.add_constraint(lhs, name_, rhs, GRB.GREATER_EQUAL, region_constraint=True))
        right_extreme_ray = cone.right_extreme_ray()
        x_coeff = math.tan(math.radians(right_extreme_ray.angle_in_degrees()))
        y_coeff = 1.0
        lhs = x_coeff * x_var - y_coeff * y_var
        rhs = 0.0
        name_ = "_".join([name, ConstraintGenerationUtilities._RIGHT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX])
        constraints.append(model.add_constraint(lhs, name_, rhs, GRB.LESS_EQUAL, region_constraint=True))
        return constraints

    @staticmethod
    def create_constraint_for_edge_in_two_dimension(model, edge, x_var, y_var, name=None, sense=GRB.GREATER_EQUAL):
        """Creates and adds the constraint to the model for the given edge, returns the constraint"""
        assert isinstance(edge, EdgeInTwoDimension)
        name = name or str(id(edge))
        name_ = "_".join([ConstraintGenerationUtilities._EDGE_CONSTRAINT_NAME_PREFIX, name])
        left_point = edge.left_point()
        right_point = edge.right_point()
        if (left_point.z1() - right_point.z1()) == 0:
            return model.add_constraint(x_var, name_, left_point.z1(), sense, region_constraint=True)
        m = (left_point.z2() - right_point.z2()) / (left_point.z1() - right_point.z1())
        x_coeff = -1 * m
        y_coeff = 1.0
        lhs = x_coeff * x_var + y_coeff * y_var   
        rhs = left_point.z2() - m * left_point.z1()
        return model.add_constraint(lhs, name_, rhs, sense, region_constraint=True)

    @staticmethod
    def create_constraints_for_lower_bound_in_two_dimension(model, lower_bound, x_var, y_var, name=None):
        """Creates and adds the constraints to the model for the given lower bound, returns the constraints"""
        assert isinstance(lower_bound, LowerBoundInTwoDimension)
        name = name or str(id(lower_bound))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z1"])
        constraints = []
        constraints.append(
            model.add_constraint(x_var, name_, lower_bound.z1(), GRB.GREATER_EQUAL, region_constraint=True))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z2"])
        if lower_bound.z2():
            constraints.append(
                model.add_constraint(y_var, name_, lower_bound.z2(), GRB.GREATER_EQUAL, region_constraint=True))
        return constraints


class ModelQueryUtilities:

    """Implements model query utilities"""

    @staticmethod
    def query_optimal_solution(model, solver_stage=None):
        """Queries the model for a feasible solution, and returns the best feasible solution if there exists any"""
        status = model.getAttr("Status")
        if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
            message = "the optimization call for the '%s' model ended with the '%s' status" % (
                    model.getAttr("ModelName"), status.value)
            if solver_stage:
                message = " ".join([message, "in the '%s' stage" % solver_stage])
            raise RuntimeError(message)
        values = []
        for obj_index in range(model.getAttr("NumObj")):
            obj = model.getObjective(index=obj_index)
            values.append(obj.getValue())
        y_bar = [var.x for var in model.getVars() if var.getAttr("VType") == "B" or var.getAttr("VType") == "I"]
        return PointSolution(Point(values), y_bar)
