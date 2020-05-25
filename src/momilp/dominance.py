"""Checks the dominance of a set of points, and eliminates the dominated points"""

from enum import Enum
import math
import numpy as np
from gurobipy import GRB, Model, QuadExpr
from time import time

from src.common.elements import Edge, EdgeInTwoDimension, FrontierInTwoDimension, Point, PointInTwoDimension
from src.molp.utilities import ModelQueryUtilities as molp_query_utilities
from src.momilp.utilities import ConstraintGenerationUtilities, TypeConversionUtilities


class DominanceModel:

    """Implements model to check the dominance of a point or an edge"""

    class ConstraintKind(Enum):

        """Represents the kind of constraints in the dominance model"""

        CHECKED_ELEMENT = 0
        DOMINATED_SPACE = 1

    _CHECKED_ELEMENT_CONSTRAINTS_NAME = "checked_element"
    _DOMINATED_SPACE_CONSTRAINTS_NAME = "dominated_space"

    def __init__(self, num_objectives):
        assert num_objectives == 2, "only two objective problems are supported currently"
        self._constraints_for_checked_element = []
        self._constraints_for_dominated_space = []
        self._model = None
        self._num_objectives = num_objectives
        self._objective_variables = []
        self._create_model()

    def _add_constraint(self, kind, lhs, rhs, sense):
        """Adds constraint to the model"""
        try:
            assert not isinstance(lhs, QuadExpr)
            assert not isinstance(rhs, QuadExpr)
        except AssertionError as error:
            message = "The constraint left-hand-side '%s' cannot be added to the linear model '%s'" % (
                lhs, self._model.getAttr("ModelName"))
            raise RuntimeError(message) from error
        if kind == DominanceModel.ConstraintKind.DOMINATED_SPACE:
            name = "_".join(
                [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, str(len(self._constraints_for_dominated_space))])
            constraint = self._model.addLConstr(lhs, sense=sense, rhs=rhs, name=name)
            self._constraints_for_dominated_space.append(constraint)
        elif kind == DominanceModel.ConstraintKind.CHECKED_ELEMENT:
            name = "_".join(
                [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, str(len(self._constraints_for_checked_element))])
            constraint = self._model.addLConstr(lhs, sense=sense, rhs=rhs, name=name)
            self._constraints_for_checked_element.append(constraint)
        self._model.update()
        return constraint

    def _create_constraint_for_dominated_space_by_edge(self, edge):
        """Creates and adds the constraint to the model for the given edge, returns the constraint"""
        left_point = edge.left_point()
        right_point = edge.right_point()
        x_var = self._objective_variables[0]
        y_var = self._objective_variables[1]
        sense = GRB.LESS_EQUAL
        kind = DominanceModel.ConstraintKind.DOMINATED_SPACE
        if (left_point.z1() - right_point.z1()) == 0:
            return self._add_constraint(kind, x_var, left_point.z1(), sense)
        m = (left_point.z2() - right_point.z2()) / (left_point.z1() - right_point.z1())
        x_coeff = -1 * m
        y_coeff = 1.0
        lhs = x_coeff * x_var + y_coeff * y_var   
        rhs = left_point.z2() - m * left_point.z1()
        return self._add_constraint(kind, lhs, rhs, sense)

    def _create_model(self):
        """Creates the model"""
        model = Model("dominance_model")
        model.setAttr("ModelSense", GRB.MAXIMIZE)
        num_objectives = self._num_objectives
        z = model.addVars(num_objectives, name="z")
        for i in range(num_objectives):
            model.setObjectiveN(z[i], i, priority=num_objectives - i)
        self._model = model
        self._objective_variables = z.values()
        model.setParam("LogToConsole", 0)
        model.setParam("LogFile", "")
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9

    def add_dominated_space_constraints(self, frontier):
        """Adds the constraints to the model that define the dominated space by the frontier"""
        kind = DominanceModel.ConstraintKind.DOMINATED_SPACE
        if frontier.singleton():
            point = frontier.point()
            for index, value in enumerate(point.values()):
                objective_variable = self._objective_variables[index]
                self._add_constraint(kind, objective_variable, value, GRB.LESS_EQUAL)
            return
        edges = frontier.edges()
        # first create the constraints that define the dominated space in the two-dimension
        # put a constraint on the first objective
        south_east_point = edges[-1].right_point()
        criterion_index = 0        
        objective_variable = self._objective_variables[criterion_index]
        self._add_constraint(
            kind, objective_variable, south_east_point.values()[criterion_index], GRB.LESS_EQUAL)
        # put a constraint on the second objective
        north_west_point = edges[0].left_point()
        criterion_index = 1
        objective_variable = self._objective_variables[criterion_index]
        self._add_constraint(
            kind, objective_variable, north_west_point.values()[criterion_index], GRB.LESS_EQUAL)
        # put constraints for the left half-spaces defined by the edges
        for edge in edges:
            constraint = self._create_constraint_for_dominated_space_by_edge(edge)
            self._constraints_for_dominated_space.append(constraint)

    def add_edge_constraint(self, edge):
        """Adds the constraint to restrict the feasible space to the given edge"""
        kind = DominanceModel.ConstraintKind.CHECKED_ELEMENT
        # restrict the feasible space to the edge
        left_point = edge.left_point()
        right_point = edge.right_point()
        lambda_var = self._model.addVar(name="lambda", ub=1.0)
        # add the convex combination constraint for each criterion
        for index, objective_variable in enumerate(self._objective_variables):
            self._add_constraint(
                kind, objective_variable + lambda_var * (right_point.values()[index] - left_point.values()[index]), 
                right_point.values()[index], GRB.EQUAL)

    def add_point_constraint(self, point):
        """Adds the constraint to restrict the feasible space to the given point"""
        kind = DominanceModel.ConstraintKind.CHECKED_ELEMENT
        for index, value in enumerate(point.values()):
            objective_variable = self._objective_variables[index]
            self._add_constraint(kind, objective_variable, value, GRB.EQUAL)

    def remove_checked_element_constraints(self):
        """Removes the constraints that were added to restrict the feasible space to the element"""
        self._model.remove(self._constraints_for_checked_element)
        self._constraints_for_checked_element = []

    def remove_dominated_space_constraints(self):
        """Removes the constraints that were added to define the dominated space"""
        self._model.remove(self._constraints_for_dominated_space)
        self._constraints_for_dominated_space = []

    def solve(self, objective_index=None):
        """Solves the model for the objective specified by the objective index, returns the point in the objective 
        space corresponding to the optimal solution
        
        NOTE: A lexicographic objective function is solved where the given objective index is assigned top priority"""
        model = self._model
        if objective_index is not None:
            model.setParam("ObjNumber", objective_index)
            current_priority = model.getAttr("ObjNPriority")
            top_priority = self._num_objectives
            if current_priority != top_priority:
                index_with_top_priority = None
                for index in range(self._num_objectives):
                    model.setParam("ObjNumber", index)
                    priority = model.getAttr("ObjNPriority")
                    if priority == top_priority:
                        index_with_top_priority = index
                        break
                # swap the priorities
                model.setParam("ObjNumber", objective_index)
                model.setAttr("ObjNPriority", top_priority)
                model.setParam("ObjNumber", index_with_top_priority)
                model.setAttr("ObjNPriority", current_priority)
        # solve
        model.optimize()
        values, status = molp_query_utilities.query_optimal_objective_values(model, raise_error_if_infeasible=False)
        return Point(values) if status != GRB.INFEASIBLE else None


class DominanceRules:

    """Implements rules to identify the dominance relations between two sets of points, edges or frontiers"""

    # MOMILP_TO_DO: MOMILP-10: Dominance rules for open or half-open edges
    class EdgeToEdge:

        """Implements the rules when comparing an edge to another edge"""

        @staticmethod
        def dominated(this, that):
            """Returns True if 'this' edge is dominated by 'that' edge, otherwise False"""
            assert isinstance(this, EdgeInTwoDimension) and isinstance(that, EdgeInTwoDimension), "the edges must be " \
                "in the two-dimensional space"
            if not DominanceRules.PointToEdge.dominated(this.left_point(), that):
                return False
            if not DominanceRules.PointToEdge.dominated(this.right_point(), that):
                return False            
            return True

    class EdgeToFrontier:

        """Implements the rules when comparing an edge to a frontier"""

        @staticmethod
        def dominated(edge, frontier):
            """Returns True if the edge is dominated by the frontier, otherwise False"""
            assert isinstance(edge, EdgeInTwoDimension) and isinstance(frontier, FrontierInTwoDimension), "the edge " \
                "and the frontier must be in the two-dimensional space"
            if not DominanceRules.PointToFrontier.dominated(edge.left_point(), frontier):
                return False
            if not DominanceRules.PointToFrontier.dominated(edge.right_point(), frontier):
                return False            
            return True
    
    class EdgeToPoint:

        """Implements the rules when comparing an edge to a point"""

        @staticmethod
        def dominated(edge, point):
            """Returns True if the edge is dominated by the point, otherwise False"""
            assert len(edge.start_point().values()) == len(point.values()), \
                "the start point of the edge and the compared point must have the same number of dimensions"
            start_point_dominated = all([x <= y for x, y in zip(edge.start_point().values(), point.values())])
            assert len(edge.end_point().values()) == len(point.values()), \
                "the end point of the edge and the compared point must have the same number of dimensions"
            end_point_dominated = all([x <= y for x, y in zip(edge.end_point().values(), point.values())])
            return start_point_dominated and end_point_dominated

    class FrontierToEdge:

        """Implements the rules when comparing a frontier to an edge"""

        @staticmethod
        def dominated(frontier, edge):
            """Returns True if the frontier is dominated by the edge"""
            assert isinstance(frontier, FrontierInTwoDimension), "the frontier must be in two-dimensional space"
            assert isinstance(edge, EdgeInTwoDimension), "the edge must be in two-dimensional space"
            if frontier.singleton():
                return DominanceRules.PointToEdge.dominated(frontier.point(), edge)
            for frontier_edge in frontier.edges():
                if not DominanceRules.EdgeToEdge.dominated(frontier_edge, edge):
                    return False
            return True

    class FrontierToFrontier:

        """Implements the rules when comparing a frontier to another one"""

        @staticmethod
        def dominated(this, that):
            """Returns True if 'this' frontier is dominated by 'that' frontier"""
            assert isinstance(this, FrontierInTwoDimension) and isinstance(that, FrontierInTwoDimension), "the " \
                "frontiers must be in two-dimensional space"
            if this.point():
                return DominanceRules.PointToFrontier.dominated(this.point(), that)
            for edge in this.edges():
                if not DominanceRules.EdgeToFrontier.dominated(edge, that):
                    return False
            return True

    class FrontierToPoint:

        """Implements the rules when comparing a frontier to a point"""

        @staticmethod
        def dominated(frontier, point):
            """Returns True if the frontier is dominated by the point, otherwise False"""
            if frontier.singleton():
                return DominanceRules.PointToPoint.dominated(frontier.point(), point)
            if len(frontier.edges()) == 1:
                return DominanceRules.EdgeToPoint.dominated(frontier.edges()[0], point)
            return DominanceRules.EdgeToPoint.dominated(frontier.edges()[0], point) and \
                DominanceRules.EdgeToPoint.dominated(frontier.edges()[-1], point)
    
    class PointToEdge:

        """Implements the rules when comparing a point to an edge"""

        @staticmethod
        def dominated(point, edge):
            """Returns True if the point is dominated by the edge, otherwise False"""
            assert isinstance(point, PointInTwoDimension) and isinstance(edge, EdgeInTwoDimension), \
                "this method is only available for points and edges in two-dimensional space"
            if point.z2() > edge.left_point().z2():
                return False
            if point.z1() > edge.right_point().z1():
                return False
            return not (np.dot(point.values(), edge.normal_vector()) >= edge.edge_value())

    class PointToEdgeSet:

        """Implements the rules when comparing a point to a set of edges"""

        @staticmethod
        def dominated(point, edges, edge_dimensions_with_constant_value=None):
            """Returns True if the point is dominated by any edge, otherwise False"""
            if not edges:
                return False
            assert isinstance(point, Point)
            p_dim = len(point.values())
            assert all([isinstance(e, Edge) and len(e.start_point().values()) == p_dim for e in edges])
            edge_dimensions_with_constant_value = edge_dimensions_with_constant_value or []
            edge_dimensions = list(
                set(range(len(edges[0].start_point().values()))) - set(edge_dimensions_with_constant_value))
            assert len(edge_dimensions) == 2, "dimension of edges must be at exactly 2"
            for edge in edges:
                if any(
                        [point.values()[d] > edge.start_point().values()[d] for d in 
                         edge_dimensions_with_constant_value]):
                    continue
                edge_in_two_dimension = TypeConversionUtilities.edge_to_edge_in_two_dimension(edge_dimensions, edge)
                point_in_two_dimension = TypeConversionUtilities.point_to_point_in_two_dimension(edge_dimensions, point)
                if DominanceRules.PointToEdge.dominated(point_in_two_dimension, edge_in_two_dimension):
                    return True
            return False

        @staticmethod
        def dominated_in_two_dimension(point, edges):
            """Returns True if the point is dominated by any edge, otherwise False"""
            assert isinstance(point, PointInTwoDimension)
            assert all([isinstance(e, EdgeInTwoDimension) for e in edges])
            for edge in edges:
                if DominanceRules.PointToEdge.dominated(point, edge):
                    return True
            return False

    class PointToFrontier:

        """Implements the rules when comparing a point to a frontier"""

        @staticmethod
        def dominated(point, frontier):
            """Returns True if the point is dominated by the frontier, otherwise False"""
            assert isinstance(point, PointInTwoDimension) and isinstance(frontier, FrontierInTwoDimension), \
                "this method is only available for points and frontiers in two-dimensional space"
            frontier_north_west_point = frontier.point() or frontier.edges()[0].left_point()
            if point.z2() > frontier_north_west_point.z2():
                return False
            frontier_south_east_point = frontier.point() or frontier.edges()[-1].right_point()
            if point.z1() > frontier_south_east_point.z1():
                return False
            for edge in frontier.edges():
                if np.dot(point.values(), edge.normal_vector()) >= edge.edge_value():
                    return False
            return True

    class PointToPoint:

        """Implements the rules when comparing a point to another point"""

        @staticmethod
        def dominated(this, that):
            """Returns True if 'this' point is dominated by 'that' point, otherwise False"""
            assert len(this.values()) == len(that.values()), \
                "the compared points must have the same number of dimensions"
            return this != that and all([v <= that.values()[i] for i, v in enumerate(this.values())])


    class PointToPointSet:

        """Implements the rules when comparing a point to a set of points"""

        @staticmethod
        def dominated(this_point, other_points):
            """Returns True if 'this_point' is dominated by 'other_points', otherwise False"""
            for point in other_points:
                if DominanceRules.PointToPoint.dominated(this_point, point):
                    return True
            return False


class ModelBasedDominanceFilter:

    """Filters the points that are relatively nondominated with respect to the points or edges checked against"""

    def __init__(self, num_objectives):
        assert num_objectives == 2, "only two objective problems are supported currently"
        self._dominance_model = DominanceModel(num_objectives)
        self._elapsed_time_in_seconds = 0
        self._num_models_solved = 0

    def _solve_model(self, objective_index=None):
        """Maximizes the selected criterion in the model, and returns the point corresponding to the optimal solution"""
        self._num_models_solved += 1
        start = time()
        result = self._dominance_model.solve(objective_index=objective_index)
        end = time()
        self._elapsed_time_in_seconds += end - start
        return result

    def _reset_model(self):
        """Removes the constraints related to a previous dominance check if there exist any"""
        self._dominance_model.remove_dominated_space_constraints()
        self._dominance_model.remove_checked_element_constraints()

    def elapsed_time_in_seconds(self):
        """Returns the elapsed time in seconds"""
        return self._elapsed_time_in_seconds

    def filter_edge(self, edge, tol=1e-6):
        """Filters the dominated points in the edge, and returns the updated edges"""
        self._dominance_model.remove_checked_element_constraints()
        self._dominance_model.add_edge_constraint(edge)
        # solve for z1
        z1_objective_index = 0
        point_z_1_star = self._solve_model(objective_index=z1_objective_index)
        if point_z_1_star is None:
            # the model is infeasible meaning that the edge is nondominated
            return [edge]
        # solve for z2
        z2_objective_index = 1
        point_z_2_star = self._solve_model(objective_index=z2_objective_index)
        # MOMILP_TO_DO: Check if a point of the edge is on the boundary of the dominated space
        if math.isclose(
                point_z_1_star.values()[z1_objective_index], edge.right_point().values()[z1_objective_index], 
                rel_tol=tol):
            if math.isclose(
                    point_z_2_star.values()[z2_objective_index], edge.left_point().values()[z2_objective_index], 
                    rel_tol=tol):
                # the edge is dominated
                return []
            else:
                # the edge is partially nondominated
                values_on_the_boundary = [
                    point_z_2_star.values()[z1_objective_index], point_z_2_star.values()[z2_objective_index]]
                right_point = PointInTwoDimension(values_on_the_boundary)
                return [EdgeInTwoDimension(edge.left_point(), right_point, right_inclusive=False, z3=edge.z3())]
        else:
            if math.isclose(
                    point_z_2_star.values()[z2_objective_index], edge.left_point().values()[z2_objective_index], 
                    rel_tol=tol):
                values_on_the_boundary = [
                    point_z_1_star.values()[z1_objective_index], point_z_1_star.values()[z2_objective_index]]
                left_point = PointInTwoDimension(values_on_the_boundary)
                return [EdgeInTwoDimension(left_point, edge.right_point(), left_inclusive=False, z3=edge.z3())]
            else:
                inner_left_point = PointInTwoDimension(
                    [point_z_2_star.values()[z1_objective_index], point_z_2_star.values()[z2_objective_index]])
                inner_right_point = PointInTwoDimension(
                    [point_z_1_star.values()[z1_objective_index], point_z_1_star.values()[z2_objective_index]])
                return [
                    EdgeInTwoDimension(edge.left_point(), inner_left_point, right_inclusive=False, z3=edge.z3()), 
                    EdgeInTwoDimension(inner_right_point, edge.right_point(), left_inclusive=False, z3=edge.z3())]

    def filter_point(self, point):
        """Returns the point if it is not dominated, otherwise None"""
        self._dominance_model.remove_checked_element_constraints()
        self._dominance_model.add_point_constraint(point)
        dominated_point = self._solve_model()
        return point if not dominated_point else None

    def num_models_solved(self):
        """Returns the number of models solved"""
        return self._num_models_solved

    def set_dominated_space(self, frontier, reset=True):
        """Sets the dominated space in the model"""
        if reset:
            self._reset_model()
        dominance_model = self._dominance_model
        dominance_model.add_dominated_space_constraints(frontier)
