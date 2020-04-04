"""Implements the utilities for the momilp solver"""

from copy import deepcopy
from gurobipy import Constr, GRB, LinExpr
import math
import numpy as np
import operator
import os
import pandas as pd
from src.common.elements import point_on_ray_in_two_dimension, ConvexConeInPositiveQuadrant, Edge, EdgeInTwoDimension, \
    FrontierEdgeInTwoDimension, FrontierInTwoDimension, LowerBoundInTwoDimension, OptimizationStatus, Point, \
    PointInTwoDimension, PointSolution, RayInTwoDimension, SearchProblemResult, SearchRegionInTwoDimension


class ConstraintGenerationUtilities:

    """Implements constraint generator utility"""

    _EDGE_CONSTRAINT_NAME_PREFIX= "edge"
    _LEFT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "left_extr_ray"
    _LOWER_BOUND_CONSTRAINT_NAME_PREFIX = "lb"
    _RIGHT_EXTREME_RAY_CONSTRAINT_NAME_SUFFIX = "right_extr_ray"

    @staticmethod
    def create_constraints_for_cone_in_positive_quadrant(momilp_model, cone, x_var, y_var, name=None):
        """Creates and adds the constraints to the model for the given cone, returns the constraints"""
        assert isinstance(cone, ConvexConeInPositiveQuadrant), "incorrect cone type"
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
        assert isinstance(edge, EdgeInTwoDimension), "incorrect edge type"
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
        assert isinstance(lower_bound, LowerBoundInTwoDimension), "incorrect lower bound type"
        name = name or str(id(lower_bound))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z1"])
        constraints = []
        if lower_bound.z1():
            constraints.append(
                momilp_model.add_constraint(x_var, name_, lower_bound.z1(), GRB.GREATER_EQUAL, region_constraint=True))
        name_ = "_".join([ConstraintGenerationUtilities._LOWER_BOUND_CONSTRAINT_NAME_PREFIX, name, "z2"])
        if lower_bound.z2():
            constraints.append(
                momilp_model.add_constraint(y_var, name_, lower_bound.z2(), GRB.GREATER_EQUAL, region_constraint=True))
        return constraints

    @staticmethod
    def create_tabu_constraint(momilp_model, name, y_bar):
        """Creates and adds the tabu-constraints to the model to exclude the integer y-vectors from the feasible set"""
        model = momilp_model.problem()
        y = momilp_model.y()        
        tabu_constraint_lhs = LinExpr()
        tabu_constraint_rhs = 1
        for index, (y_, y_bar_) in enumerate(zip(y, y_bar)):
            if y_bar_ == y_.LB:
                tabu_constraint_lhs.add(y_)
                tabu_constraint_rhs += y_bar_
            elif y_bar_ == y_.UB:
                tabu_constraint_lhs.add(y_, -1)
                tabu_constraint_rhs -= y_bar_
            else:
                big_m = (y_.UB - y_.LB) if y_.UB < GRB.INFINITY else model.Params.FeasRelaxBigM
                y_pos = model.addVar(ub=big_m, vtype=GRB.INTEGER)
                y_neg = model.addVar(ub=big_m, vtype=GRB.INTEGER)
                tabu_constraint_lhs.addTerms([1, 1], [y_pos, y_neg])
                deviation_lhs = LinExpr()
                deviation_lhs.addTerms([1, -1, 1], [y_, y_pos, y_neg])
                deviation_constraint_name = "_".join([name, str(index), "dev"])
                momilp_model.add_constraint(
                    deviation_lhs, deviation_constraint_name, y_bar_, GRB.EQUAL, tabu_constraint=True)
                beta_var = model.addVar(vtype=GRB.BINARY)
                positive_deviation_lhs = LinExpr()
                positive_deviation_lhs.addTerms([1, -big_m], [y_pos, beta_var])
                positive_deviation_constraint_name = "_".join([name, str(index), "pos_dev"])
                momilp_model.add_constraint(
                    positive_deviation_lhs, positive_deviation_constraint_name, 0.0, GRB.LESS_EQUAL, 
                    tabu_constraint=True)
                negative_deviation_lhs = LinExpr()
                negative_deviation_lhs.addTerms([1, big_m], [y_neg, beta_var])
                negative_deviation_constraint_name = "_".join([name, str(index), "neg_dev"])
                momilp_model.add_constraint(
                    negative_deviation_lhs, negative_deviation_constraint_name, big_m, GRB.LESS_EQUAL, 
                    tabu_constraint=True)
        momilp_model.add_constraint(
            tabu_constraint_lhs, name, tabu_constraint_rhs, GRB.GREATER_EQUAL, tabu_constraint=True)


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
    def query_optimal_solution(
            model, y, raise_error_if_infeasible=False, round_integer_vector_values=True, solver_stage=None):
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
        y_bar = [round(var.x) if round_integer_vector_values else var.x for var in y]
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
            item[0] for item in sorted(value_index_2_priority.items(), key=operator.itemgetter(1), reverse=True)]
        for index in indicies_in_lexicographic_order:
            if base_point.values()[index] > compared_point.values()[index]:
                return 1
            if base_point.values()[index] < compared_point.values()[index]:
                return -1
        return 0


class ReportCreator:

    """Implements the report utilities"""

    _CONNECTED_POINT_INDICATOR_COLUMN_NAME = "connected"
    _FILE_NAME_TEMPLATE = "{instance_name}_{report_name}.csv"
    _NONDOMINATED_SET_REPORT_NAME = "nds"

    def __init__(self, momilp_model, state, instance_name, output_dir):
        self._instance_name = instance_name
        self._momilp_model = momilp_model
        self._nondominated_edges_df = None
        self._nondominated_points_df = None
        self._output_dir = output_dir
        self._state = state

    def _restore_original_values(self, values):
        """Restores the original values of the criteria by using the objective inverse scalers"""
        objective_index_2_name = self._momilp_model.objective_index_2_name()
        return [
            self._momilp_model.objective_scaler(name, inverse=True)(values[index]) for index, name 
            in objective_index_2_name.items()]

    def _set_nondominated_edges_df(self):
        """Sets the data frame of the nondominated edges"""
        model_sense = self._momilp_model.model_sense()
        solution_state = self._state.solution_state()
        obj_index_2_name = self._momilp_model.objective_index_2_name()
        edges = [nondominated_edge.edge() for nondominated_edge in solution_state.nondominated_edges()]
        updated_edges = []
        for edge in edges:
            start_point_values = self._restore_original_values(edge.start_point().values())
            end_point_values = self._restore_original_values(edge.end_point().values())
            # we need to switch the start and end points in case it is a minimization problem
            start_point, end_point = (Point(start_point_values), Point(end_point_values)) if model_sense == -1 else \
                (Point(end_point_values), Point(start_point_values))
            start_inclusive, end_inclusive = (edge.start_inclusive(), edge.end_inclusive()) if model_sense == -1 else \
                (edge.end_inclusive(), edge.start_inclusive())
            updated_edge = Edge(start_point, end_point, end_inclusive=end_inclusive, start_inclusive=start_inclusive)
            updated_edges.append(updated_edge)
        sorted_edges = self._sort_nondominated_edges(updated_edges)
        records = []
        for index, edge in enumerate(sorted_edges):
            start_point_values = edge.start_point().values()
            start_point_column_name_2_value = {
                obj_index_2_name[index]: value for index, value in enumerate(start_point_values)}
            start_point_column_name_2_value[ReportCreator._CONNECTED_POINT_INDICATOR_COLUMN_NAME] = 1
            records.append(start_point_column_name_2_value)
            end_point_values = edge.end_point().values()
            end_point_column_name_2_value = {
                obj_index_2_name[index]: value for index, value in enumerate(end_point_values)}
            connected = 1 if index < len(sorted_edges) - 1 and all(np.isclose(
                end_point_values, sorted_edges[index+1].start_point().values())) else 0
            end_point_column_name_2_value[ReportCreator._CONNECTED_POINT_INDICATOR_COLUMN_NAME] = connected
            if not connected:
                records.append(end_point_column_name_2_value)
        self._nondominated_edges_df = pd.DataFrame.from_records(records) if records else \
            pd.DataFrame(columns=obj_index_2_name.values())

    def _set_nondominated_points_df(self):
        """Sets the data frame of the nondominated points"""
        solution_state = self._state.solution_state()
        obj_index_2_name = self._momilp_model.objective_index_2_name()
        points = [nondominated_point.point() for nondominated_point in solution_state.nondominated_points()]
        updated_points = []
        for point in points:
            values = self._restore_original_values(point.values())
            updated_point = Point(values)
            updated_points.append(updated_point)
        sorted_points = self._sort_nondominated_points(updated_points)
        records = []
        for point in sorted_points:
            values = point.values()
            column_name_2_value = {obj_index_2_name[index]: value for index, value in enumerate(values)}
            column_name_2_value[ReportCreator._CONNECTED_POINT_INDICATOR_COLUMN_NAME] = 0
            records.append(column_name_2_value)
        self._nondominated_points_df = pd.DataFrame.from_records(records) if records else \
            pd.DataFrame(columns=obj_index_2_name.values())

    def _sort_nondominated_edges(self, edges):
        """Sorts the nondominated edges in non-improving values of the criterion values starting from the lowest index 
        criterion, returns the sorted edges"""

        model_sense = self._momilp_model.model_sense()

        def sort_function(edge):
            """Nondominated edge sorting function"""
            return tuple([model_sense * v for v in edge.start_point().values()])
        
        return sorted(edges, key=sort_function)

    def _sort_nondominated_points(self, points):
        """Sorts the nondominated points in non-improving values of the criterion values starting from the lowest index 
        criterion, returns the sorted points"""

        model_sense = self._momilp_model.model_sense()

        def sort_function(point):
            """Nondominated point sorting function"""
            return tuple([model_sense * v for v in point.values()])
        
        return sorted(points, key=sort_function)

    def _to_csv(self, df, report_name):
        """Converts the data frame to CSV file and exports to the output directory"""
        file_name = ReportCreator._FILE_NAME_TEMPLATE.format(instance_name=self._instance_name, report_name=report_name)
        df.to_csv(os.path.join(self._output_dir, file_name))

    def create(self):
        """Creates the reports"""
        self.create_data_frames()
        nondominated_set_df = self._nondominated_points_df.append(self._nondominated_edges_df, sort=False)
        report_name = ReportCreator._NONDOMINATED_SET_REPORT_NAME
        self._to_csv(nondominated_set_df, report_name)

    def create_data_frames(self):
        """Creates the data frames of the report"""
        self._set_nondominated_edges_df()
        self._set_nondominated_points_df()

    def nondominated_edges_df(self):
        """Returns the nondominated edges data frame"""
        return self._nondominated_edges_df
        
    def nondominated_points_df(self):
        """Returns the nondominated points data frame"""
        return self._nondominated_points_df


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
    def create_search_region_in_two_dimension(x_obj_name, y_obj_name, cone, edge=None, lower_bound=None, id_=None):
        """Cretaes a search region in two dimension with a minimum number of constraints"""
        if not edge and not lower_bound:
            return SearchRegionInTwoDimension(x_obj_name, y_obj_name, cone, id_=id_)
        if not edge:
            return SearchRegionInTwoDimension(x_obj_name, y_obj_name, cone, lower_bound=lower_bound, id_=id_)
        bounds = lower_bound.bounds()
        include_edge = True
        if bounds[0] >= edge.right_point().values()[0] or bounds[1] >= edge.left_point().values()[1] or \
            np.dot(edge.normal_vector(), bounds) >= edge.edge_value():
            include_edge = False
        edge_ = edge if include_edge else None
        return SearchRegionInTwoDimension(x_obj_name, y_obj_name, cone, edge=edge_, lower_bound=lower_bound, id_=id_)

    @staticmethod
    def find_intersection_of_ray_and_edge_in_two_dimension(edge, ray):
        """Finds and returns the intersection of a ray and an edge
        
        NOTE: Raises RuntimeError if there is no intersection."""
        assert isinstance(ray, RayInTwoDimension)
        assert isinstance(edge, (EdgeInTwoDimension, FrontierEdgeInTwoDimension))
        tan_of_left_extreme_point = edge.left_point().z2() / edge.left_point().z1() if edge.left_point().z1() != 0 \
            else float("inf")
        degrees_of_left_extreme_point = math.degrees(math.atan(tan_of_left_extreme_point))
        tan_of_right_extreme_point = edge.right_point().z2() / edge.right_point().z1() if edge.right_point().z1() != 0 \
            else float("inf")
        degrees_of_right_extreme_point = math.degrees(math.atan(tan_of_right_extreme_point))
        if ray.angle_in_degrees() > degrees_of_left_extreme_point or \
                ray.angle_in_degrees() < degrees_of_right_extreme_point:
            raise RuntimeError("the '%s' ray does not intersect the '%s' edge" % (ray, edge))
        tan_of_ray = math.tan(math.radians(ray.angle_in_degrees()))
        x = edge.edge_value() / np.dot(edge.normal_vector(), [1, tan_of_ray])
        y = tan_of_ray * x
        return PointInTwoDimension([x, y])

    @staticmethod
    def partition_search_region_in_two_dimension(initial_frontier, initial_region, lower_bound_delta=0.0):
        """Partition the search region in two dimension
        
        NOTE: Eliminates the subset of the region dominated by the frontier, and returns the relatively nondominated 
        sub-regions defined by the rays passing thorugh the extreme points of the frontier. Returned regions are in the 
        order of cones with left extreme rays having non-increasing angles with the x-axis (index 0)"""
        frontier = deepcopy(initial_frontier)
        region = deepcopy(initial_region)
        assert isinstance(frontier, FrontierInTwoDimension)
        assert isinstance(region, SearchRegionInTwoDimension)
        x_obj_name = region.x_obj_name()
        y_obj_name = region.y_obj_name()
        initial_lb = region.lower_bound().bounds() if region.lower_bound() else [0, 0]
        origin = PointInTwoDimension([0, 0])
        regions = []
        if frontier.point():
            point = frontier.point()
            # only update the bounds of the region if the point is on the boundary of the convex cone
            if point_on_ray_in_two_dimension(point, region.cone().left_extreme_ray()):
                left_extreme_ray = region.cone().left_extreme_ray()
                right_extreme_ray = region.cone().right_extreme_ray()
                cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
                bounds = [point.z1() + lower_bound_delta, initial_lb[1]]
                lb = LowerBoundInTwoDimension(bounds)
                region_ = SearchUtilities.create_search_region_in_two_dimension(
                    x_obj_name, y_obj_name, cone, edge=region.edge(), lower_bound=lb)
                regions.append(region_)
            elif point_on_ray_in_two_dimension(point, region.cone().right_extreme_ray()):
                left_extreme_ray = region.cone().left_extreme_ray()
                right_extreme_ray = region.cone().right_extreme_ray()
                cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
                bounds = [initial_lb[0], point.z2() + lower_bound_delta]
                lb = LowerBoundInTwoDimension(bounds)
                region_ = SearchUtilities.create_search_region_in_two_dimension(
                    x_obj_name, y_obj_name, cone, edge=region.edge(), lower_bound=lb)
                regions.append(region_)
            else:
                middle_ray = SearchUtilities.create_ray_in_two_dimension(origin, point)
                left_edge = None
                right_edge = None
                if region.edge():
                    edge_intersection_point = SearchUtilities.find_intersection_of_ray_and_edge_in_two_dimension(
                        region.edge(), middle_ray)
                    left_edge = EdgeInTwoDimension(region.edge().left_point(), edge_intersection_point)
                    right_edge = EdgeInTwoDimension(edge_intersection_point, region.edge().right_point())
                # left cone
                left_extreme_ray = region.cone().left_extreme_ray()
                right_extreme_ray = middle_ray
                cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
                bounds = [initial_lb[0], point.z2() + lower_bound_delta]
                lb = LowerBoundInTwoDimension(bounds)
                region_ = SearchUtilities.create_search_region_in_two_dimension(
                    x_obj_name, y_obj_name, cone, edge=left_edge, lower_bound=lb)
                regions.append(region_)
                # right cone
                right_extreme_ray = region.cone().right_extreme_ray()
                left_extreme_ray = middle_ray
                cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
                bounds = [point.z1() + lower_bound_delta, initial_lb[1]]
                lb = LowerBoundInTwoDimension(bounds)
                region_ = SearchUtilities.create_search_region_in_two_dimension(
                    x_obj_name, y_obj_name, cone, edge=right_edge, lower_bound=lb)
                regions.append(region_)
            return regions
        # add a region for the left-most region if the left-most point is not on the left extreme ray of the region
        left_most_point = frontier.edges()[0].left_point()
        if not point_on_ray_in_two_dimension(left_most_point, region.cone().left_extreme_ray()):
            right_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, left_most_point)
            edge = None
            if region.edge():
                edge_intersection_point = SearchUtilities.find_intersection_of_ray_and_edge_in_two_dimension(
                    region.edge(), right_extreme_ray)
                edge = EdgeInTwoDimension(region.edge().left_point(), edge_intersection_point)
            left_most_region = SearchUtilities.create_search_region_in_two_dimension(
                x_obj_name, y_obj_name, 
                ConvexConeInPositiveQuadrant([region.cone().left_extreme_ray(), right_extreme_ray]), 
                edge=edge,
                lower_bound=LowerBoundInTwoDimension([initial_lb[0], left_most_point.z2() + lower_bound_delta]))
            regions.append(left_most_region)
        # add the corresponding region for each edge in the frontier
        for edge in frontier.edges():
            left_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, edge.left_point())
            right_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, edge.right_point())
            cone = ConvexConeInPositiveQuadrant([left_extreme_ray, right_extreme_ray])
            region_ = SearchUtilities.create_search_region_in_two_dimension(
                x_obj_name, y_obj_name, cone, edge=edge, 
                lower_bound=LowerBoundInTwoDimension([initial_lb[0], initial_lb[1]]))
            regions.append(region_)
        # add a region for the right-most region
        right_most_point = frontier.edges()[-1].right_point()
        if not point_on_ray_in_two_dimension(right_most_point, region.cone().right_extreme_ray()):
            left_extreme_ray = SearchUtilities.create_ray_in_two_dimension(origin, right_most_point)
            edge = None
            if region.edge():
                edge_intersection_point = SearchUtilities.find_intersection_of_ray_and_edge_in_two_dimension(
                    region.edge(), left_extreme_ray)
                edge = EdgeInTwoDimension(edge_intersection_point, region.edge().right_point())
            right_most_region = SearchUtilities.create_search_region_in_two_dimension(
                x_obj_name, y_obj_name, 
                ConvexConeInPositiveQuadrant([left_extreme_ray, region.cone().right_extreme_ray()]), 
                edge=edge, 
                lower_bound=LowerBoundInTwoDimension([right_most_point.z1() + lower_bound_delta, initial_lb[1]]))
            regions.append(right_most_region)
        return regions


class TypeConversionUtilities:

    """Implements the utilities to convert types to each other"""

    @staticmethod
    def edge_in_two_dimension_to_edge(additional_dim_2_value, edge_in_two_dimension, projected_space_dim_2_dim):
        """Returns an edge in the original search space from the edge in the projected search space"""
        start_point = {
            projected_space_dim_2_dim[index]: value for index, value in 
            enumerate(edge_in_two_dimension.left_point().values())}
        start_point.update(additional_dim_2_value)
        start_point_values = [start_point[key] for key in sorted(start_point)]
        end_point = {
            projected_space_dim_2_dim[index]: value for index, value in 
            enumerate(edge_in_two_dimension.right_point().values())}
        end_point.update(additional_dim_2_value)
        end_point_values = [end_point[key] for key in sorted(end_point)]
        return Edge(Point(start_point_values), Point(end_point_values))

    @staticmethod
    def edge_to_edge_in_two_dimension(dimensions, edge):
        """Converts the edge to the edge in two dimensional space of the specified dimensions"""
        assert len(dimensions) == 2, "there must be two dimensions"
        start_point_at_left = edge.start_point().values()[dimensions[0]] <= edge.end_point().values()[dimensions[0]]
        left_point, right_point = (edge.start_point(), edge.end_point()) if start_point_at_left else \
            (edge.end_point(), edge.start_point())
        left_point_in_two_dimension = PointInTwoDimension(
            [value for index, value in enumerate(left_point.values()) if index in dimensions])
        right_point_in_two_dimension = PointInTwoDimension(
            [value for index, value in enumerate(right_point.values()) if index in dimensions])
        left_inclusive, right_inclusive = (edge.start_inclusive(), edge.end_inclusive()) if start_point_at_left else \
            (edge.end_inclusive(), edge.start_inclusive())
        return EdgeInTwoDimension(
            left_point_in_two_dimension, right_point_in_two_dimension, left_inclusive=left_inclusive, 
            right_inclusive=right_inclusive)

    @staticmethod
    def point_to_point_in_two_dimension(dimensions, point):
        """Converts the point to point in two dimensional space of the specified dimensions"""
        assert len(dimensions) == 2, "there must be two dimensions"
        sorted_dimensions = sorted(dimensions)
        return PointInTwoDimension([value for index, value in enumerate(point.values()) if index in sorted_dimensions])