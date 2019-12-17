"""Implements the utilities for the momilp solver"""

from gurobipy import Constr, GRB, LinExpr
import math
import operator
from src.common.elements import ConvexConeInPositiveQuadrant, EdgeInTwoDimension, FrontierInTwoDimension, \
    LowerBoundInTwoDimension, OptimizationStatus, Point, PointInTwoDimension, PointSolution, RayInTwoDimension, \
    SearchProblemResult, SearchRegionInTwoDimension


class ConstraintGenerationUtilities:

    """Implements constraint generator utility"""

    _EDGE_CONSTRAINT_NAME_PREFIX= "edge"
    _LEFT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "left_extr_ray"
    _LOWER_BOUND_CONSTRAINT_NAME_PREFIX = "lb"
    _RIGHT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "right_extr_ray"

    @staticmethod
    def create_binary_tabu_constraint(momilp_model, name, y_bar):
        """Creates and adds the tabu-constraints to the model to exclude the binary y-vectors from the feasible set"""
        y = momilp_model.y()        
        lhs = LinExpr()
        rhs = 1
        for y_, y_bar_ in zip(y, y_bar):
            coeff = -1 if y_bar_ == 1 else 1
            rhs = rhs - 1 if y_bar_ else rhs
            lhs.add(y_, coeff)
        momilp_model.add_constraint(lhs, name, rhs, GRB.GREATER_EQUAL, tabu_constraint=True)

    @staticmethod
    def create_constraints_for_cone_in_positive_quadrant(momilp_model, cone, x_var, y_var, name=None):
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
        constraints.append(momilp_model.add_constraint(lhs, name_, rhs, GRB.GREATER_EQUAL, region_constraint=True))
        right_extreme_ray = cone.right_extreme_ray()
        x_coeff = math.tan(math.radians(right_extreme_ray.angle_in_degrees()))
        y_coeff = 1.0
        lhs = x_coeff * x_var - y_coeff * y_var
        rhs = 0.0
        name_ = "_".join([name, ConstraintGenerationUtilities._RIGHT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX])
        constraints.append(momilp_model.add_constraint(lhs, name_, rhs, GRB.LESS_EQUAL, region_constraint=True))
        return constraints

    @staticmethod
    def create_constraint_for_edge_in_two_dimension(
            momilp_model, edge, x_var, y_var, name=None, sense=GRB.GREATER_EQUAL):
        """Creates and adds the constraint to the model for the given edge, returns the constraint"""
        assert isinstance(edge, EdgeInTwoDimension)
        name = name or str(id(edge))
        name_ = "_".join([ConstraintGenerationUtilities._EDGE_CONSTRAINT_NAME_PREFIX, name])
        left_point = edge.left_point()
        right_point = edge.right_point()
        if (left_point.z1() - right_point.z1()) == 0:
            return momilp_model.add_constraint(x_var, name_, left_point.z1(), sense, region_constraint=True)
        m = (left_point.z2() - right_point.z2()) / (left_point.z1() - right_point.z1())
        x_coeff = -1 * m
        y_coeff = 1.0
        lhs = x_coeff * x_var + y_coeff * y_var   
        rhs = left_point.z2() - m * left_point.z1()
        return momilp_model.add_constraint(lhs, name_, rhs, sense, region_constraint=True)

    @staticmethod
    def create_constraints_for_lower_bound_in_two_dimension(momilp_model, lower_bound, x_var, y_var, name=None):
        """Creates and adds the constraints to the model for the given lower bound, returns the constraints"""
        assert isinstance(lower_bound, LowerBoundInTwoDimension)
        name = name or str(id(lower_bound))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z1"])
        constraints = []
        constraints.append(
            momilp_model.add_constraint(x_var, name_, lower_bound.z1(), GRB.GREATER_EQUAL, region_constraint=True))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z2"])
        if lower_bound.z2():
            constraints.append(
                momilp_model.add_constraint(y_var, name_, lower_bound.z2(), GRB.GREATER_EQUAL, region_constraint=True))
        return constraints

    @staticmethod
    def create_integer_tabu_constraint(momilp_model, name, y_bar):
        """Creates and adds the tabu-constraints to the model to exclude the integer y-vectors from the feasible set"""
        raise NotImplementedError()


class ModelQueryUtilities:

    """Implements model query utilities"""

    GUROBI_STATUS_2_OPTIMIZATION_STATUS = {
        GRB.INF_OR_UNBD: OptimizationStatus.UNDEFINED,
        GRB.INFEASIBLE: OptimizationStatus.INFEASIBLE,
        GRB.UNBOUNDED: OptimizationStatus.UNDEFINED,
        GRB.OPTIMAL: OptimizationStatus.OPTIMAL,
        GRB.ITERATION_LIMIT: OptimizationStatus.FEASIBLE,
        GRB.NODE_LIMIT: OptimizationStatus.FEASIBLE,
        GRB.SOLUTION_LIMIT: OptimizationStatus.FEASIBLE,
        GRB.TIME_LIMIT: OptimizationStatus.FEASIBLE
    }

    @staticmethod
    def query_optimal_solution(model, raise_error_if_infeasible=False, solver_stage=None):
        """Queries the model for a feasible solution, and returns the best feasible solution if there exists any"""
        status = ModelQueryUtilities.GUROBI_STATUS_2_OPTIMIZATION_STATUS.get(
            model.getAttr("Status"), OptimizationStatus.UNDEFINED)
        error_message = "the optimization call for the '%s' model ended with the '%s' status" % (
            model.getAttr("ModelName"), status.value)
        if solver_stage:
            error_message = " ".join([error_message, "in the '%s' stage" % solver_stage])
        point_solution = None
        if status == OptimizationStatus.UNDEFINED:
            raise RuntimeError(error_message)
        if status == OptimizationStatus.INFEASIBLE:
            if raise_error_if_infeasible:
                raise RuntimeError(error_message)
            return SearchProblemResult(point_solution, status)
        values = []
        for obj_index in range(model.getAttr("NumObj")):
            obj = model.getObjective(index=obj_index)
            values.append(obj.getValue())
        y_bar = [var.x for var in model.getVars() if var.getAttr("VType") == "B" or var.getAttr("VType") == "I"]
        return SearchProblemResult(PointSolution(Point(values), y_bar), status)


class PointComparisonUtilities:

    """Implements point comparison utilities"""

    @staticmethod
    def compare_to(base_point, compared_point, value_index_2_priority):
        """Compares the points in lexicographic order specified by value index to priority dictionary
        
        Returns 0, if both points have equal values, 1 if base point is lexicographically greater than the compared 
        point, else -1"""
        assert len(base_point.values()) == len(compared_point.values())
        indicies_in_lexicographic_order = [
            operator.itemgetter(0) for item in sorted(value_index_2_priority.items(), key=operator.itemgetter(1), 
            reverse=True)]
        for index in indicies_in_lexicographic_order:
            if base_point.values()[index] > compared_point.values()[index]:
                return 1
            if base_point.values()[index] < compared_point.values()[index]:
                return -1
        return 0


class SearchUtilities:

    """Implements search utilities"""

    @staticmethod
    def create_ray_in_two_dimension(from_point, to_point):
        """Returns a ray defined by the two points"""
        assert isinstance(from_point, PointInTwoDimension)
        assert isinstance(to_point, PointInTwoDimension)
        tan = (to_point.z2() - from_point.z2()) / (to_point.z1() - from_point.z1()) if \
            to_point.z1() != from_point.z1() else float("inf")
        return RayInTwoDimension(math.degrees(math.atan(tan)), from_point)

    @staticmethod
    def partition_search_region_in_two_dimension(frontier, region):
        """Partition the search region in two dimension
        
        NOTE: Eliminates the subset of the region dominated by the frontier, and returns the relatively nondominated 
        sub-regions defined by the rays passing thorugh the extreme points of the frontier"""
        assert isinstance(frontier, FrontierInTwoDimension)
        assert isinstance(region, SearchRegionInTwoDimension)
        initial_lb = region.lower_bound() or [0, 0]
        origin = PointInTwoDimension([0, 0])
        regions = []
        if frontier.point():
            point = frontier.point()
            # left cone
            left_extreme_ray = region.cone().left_extreme_ray()
            right_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, point)
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            bounds = [initial_lb[0], point.z2()]
            lb = LowerBoundInTwoDimension(bounds)
            regions.append(SearchRegionInTwoDimension(cone, edge=region.edge(), lower_bound=lb))
            # right cone
            right_extreme_ray = region.cone().right_extreme_ray()
            left_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, point)
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            bounds = [point.z1(), initial_lb[1]]
            lb = LowerBoundInTwoDimension(bounds)
            regions.append(SearchRegionInTwoDimension(cone, edge=region.edge(), lower_bound=lb))
            return regions
        for index, edge in enumerate(frontier.edges()):
            left_extreme_ray = region.cone().left_extreme_ray() if index == 0 else \
                SearchUtilities.create_ray_in_two_dimension(origin, edge.left_point())
            right_extreme_ray = region.cone().right_extreme_ray() if index == len(frontier.edges()) - 1 else \
                SearchUtilities.create_ray_in_two_dimension(origin, edge.right_point())
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            regions.append(SearchRegionInTwoDimension(cone, edge=edge, lower_bound=initial_lb))
        return regions
