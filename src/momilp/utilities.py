"""Implements the utilities for the momilp solver"""

from gurobipy import Constr, GRB, LinExpr
import math
import operator
import os
import pandas as pd
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


class ReportCreator:

    """Implements the report utilities"""

    def __init__(self, momilp_model, state, output_dir):
        self._momilp_model = momilp_model
        self._output_dir = output_dir
        self._nondominated_edges_df = None
        self._nondominated_points_df = None
        self._state = state

    def _set_nondominated_edges_df(self):
        """Sets the nondominated edges data frame"""
        solution_state = self._state.solution_state()
        obj_index_2_name = self._momilp_model.objective_index_2_name()
        edges = [nondominated_edge.edge() for nondominated_edge in solution_state.nondominated_edges()]
        records = []
        for edge in edges:
            records.append(
                {
                    obj_index_2_name[index]: value for index, value in 
                    enumerate(zip(edge.start_point().values(), edge.end_point().values()))})
        self._nondominated_edges_df = pd.DataFrame.from_records(records)

    def _set_nondominated_points_df(self):
        """Sets the nondominated points data frame"""
        solution_state = self._state.solution_state()
        obj_index_2_name = self._momilp_model.objective_index_2_name()
        points = [nondominated_point.point() for nondominated_point in solution_state.nondominated_points()]
        records = []
        for point in points:
            records.append({obj_index_2_name[index]: value for index, value in enumerate(point.values())})
        self._nondominated_points_df = pd.DataFrame.from_records(records)

    def create_data_frames(self):
        """Creates the data frames of the report"""
        self._set_nondominated_edges_df()
        self._set_nondominated_points_df()

    def export(self):
        """Writes the files to the spcified output directory"""
        pass

    def nondominated_edges_df(self):
        """Returns the nondominated edges data frame"""
        return self._nondominated_edges_df
        
    def nondominated_points_df(self):
        """Returns the nondominated points data frame"""
        return self._nondominated_points_df


class PointComparisonUtilities:

    """Implements point comparison utilities"""

    @staticmethod
    def compare_to(base_point, compared_point, value_index_2_priority):
        """Compares the points in lexicographic order specified by value index to priority dictionary
        
        Returns 0, if both points have equal values, 1 if base point is lexicographically greater than the compared 
        point, else -1"""
        assert len(base_point.values()) == len(compared_point.values())
        indicies_in_lexicographic_order = [
            item[0] for item in sorted(value_index_2_priority.items(), key=operator.itemgetter(1), reverse=True)]
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
    def partition_search_region_in_two_dimension(frontier, region, lower_bound_delta=0.0):
        """Partition the search region in two dimension
        
        NOTE: Eliminates the subset of the region dominated by the frontier, and returns the relatively nondominated 
        sub-regions defined by the rays passing thorugh the extreme points of the frontier. Returned regions are in the 
        order of cones with left extreme rays having non-increasing angles with the x-axis (index 0)"""
        assert isinstance(frontier, FrontierInTwoDimension)
        assert isinstance(region, SearchRegionInTwoDimension)
        x_obj_name = region.x_obj_name()
        y_obj_name = region.y_obj_name()
        initial_lb = region.lower_bound().bounds() if region.lower_bound() else [0, 0]
        origin = PointInTwoDimension([0, 0])
        regions = []
        if frontier.point():
            point = frontier.point()
            # left cone
            left_extreme_ray = region.cone().left_extreme_ray()
            right_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, point)
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            bounds = [initial_lb[0], point.z2() + lower_bound_delta]
            lb = LowerBoundInTwoDimension(bounds)
            regions.append(SearchRegionInTwoDimension(x_obj_name, y_obj_name, cone, edge=region.edge(), lower_bound=lb))
            # right cone
            right_extreme_ray = region.cone().right_extreme_ray()
            left_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, point)
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            bounds = [point.z1() + lower_bound_delta, initial_lb[1]]
            lb = LowerBoundInTwoDimension(bounds)
            regions.append(SearchRegionInTwoDimension(x_obj_name, y_obj_name, cone, edge=region.edge(), lower_bound=lb))
            return regions
        for index, edge in enumerate(frontier.edges()):
            left_extreme_ray = region.cone().left_extreme_ray() if index == 0 else \
                SearchUtilities.create_ray_in_two_dimension(origin, edge.left_point())
            right_extreme_ray = region.cone().right_extreme_ray() if index == len(frontier.edges()) - 1 else \
                SearchUtilities.create_ray_in_two_dimension(origin, edge.right_point())
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            regions.append(
                SearchRegionInTwoDimension(
                    x_obj_name, y_obj_name, cone, edge=edge, lower_bound=LowerBoundInTwoDimension(initial_lb)))
        return regions
