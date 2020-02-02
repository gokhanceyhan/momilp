"""Checks the dominance of a set of points, and eliminates the dominated points"""

import numpy as np
from gurobipy import GRB
from src.common.elements import EdgeInTwoDimension, FrontierInTwoDimension, Point, PointInTwoDimension
from src.molp.utilities import ModelQueryUtilities as molp_query_utilities
from src.momilp.utilities import ConstraintGenerationUtilities


class DominanceModel:

    """Implements model to check the dominance of a point or an edge"""

    _CHECKED_ELEMENT_CONSTRAINTS_NAME = "checked_element"
    _DOMINATED_SPACE_CONSTRAINTS_NAME = "dominated_space"

    def __init__(self, momilp_model, primary_objective_index, projected_space_criterion_index_2_criterion_index):
        assert momilp_model.num_obj() == 3, "only three objective problems are supported currently"
        self._momilp_model = momilp_model
        self._num_constraints_for_checked_element = 0
        self._num_constraints_for_dominated_space = 0
        self._primary_objective_index = primary_objective_index
        self._projected_space_criterion_index_2_criterion_index = projected_space_criterion_index_2_criterion_index

    def _projected_space_criterion_index_2_objective_variable(self, projected_space_criterion_index):
        """Returns the objective variable at the specified projected space criterion index"""
        objective_index = self._projected_space_criterion_index_2_criterion_index[projected_space_criterion_index]
        objective_name = self._momilp_model.objective_index_2_name()[objective_index]
        return self._momilp_model.Z()[objective_name]

    def add_dominated_space_constraints(self, frontier, primary_objective_value):
        """Adds the constraints to the momilp model that define the dominated space by the frontier in two dimension
        
        NOTE: The frontier must be in two-dimensional projected objective space"""
        assert isinstance(frontier, FrontierInTwoDimension), \
            "the frontier must be an instance of 'FrontierInTwoDimension'"
        momilp_model = self._momilp_model
        if frontier.singleton():
            point = frontier.point()
            for index, value in enumerate(point.values()):
                objective_variable = self._projected_space_criterion_index_2_objective_variable(index)
                constraint_name = "_".join(
                    [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, self._num_constraints_for_dominated_space])
                momilp_model.add_constraint(objective_variable, constraint_name, value, GRB.LESS_EQUAL)
                self._num_constraints_for_dominated_space =+ 1
            return
        edges = frontier.edges()
        # first create the constraints that define the dominated space in the two-dimension
        north_west_point = edges[0].left_point()
        # put a constraint on the objective at the second index of the projected space
        criterion_index = 1
        objective_variable = self._projected_space_criterion_index_2_objective_variable(criterion_index)
        constraint_name = "_".join(
            [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, self._num_constraints_for_dominated_space])
        momilp_model.add_constraint(
            objective_variable, constraint_name, north_west_point.values()[criterion_index], GRB.LESS_EQUAL)
        self._num_constraints_for_dominated_space =+ 1
        south_east_point = edges[-1].right_point()
        # put a constraint on the objective at the first index of the projected space
        criterion_index = 0        
        objective_variable = self._projected_space_criterion_index_2_objective_variable(criterion_index)
        constraint_name = "_".join(
            [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, self._num_constraints_for_dominated_space])
        momilp_model.add_constraint(
            objective_variable, constraint_name, south_east_point.values()[criterion_index], GRB.LESS_EQUAL)
        self._num_constraints_for_dominated_space =+ 1
        # put constraints for the left half-spaces defined by the edges
        for edge in edges:
            x_var = self._projected_space_criterion_index_2_objective_variable(0)
            y_var = self._projected_space_criterion_index_2_objective_variable(1)
            constraint_name = "_".join(
                [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, self._num_constraints_for_dominated_space])
            ConstraintGenerationUtilities.create_constraint_for_edge_in_two_dimension(
                momilp_model, edge, x_var, y_var, name=constraint_name, sense=GRB.LESS_EQUAL)
            self._num_constraints_for_dominated_space += 1
        # add a constraint for the primary objective index
        objective_name = momilp_model.objective_index_2_name()[self._primary_objective_index]
        objective_variable = momilp_model.objective_name_2_variable()[objective_name]
        constraint_name = "_".join(
            [DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, self._num_constraints_for_dominated_space])
        momilp_model.add_constraint(objective_variable, constraint_name, primary_objective_value, GRB.LESS_EQUAL)
        momilp_model.update_model()

    def add_edge_constraint(self, edge):
        """Adds the constraint to restrict the feasible space to the given edge"""
        momilp_model = self._momilp_model
        # first fix the value in the primary objective index
        objective_value = edge.z3()
        objective_name = momilp_model.objective_index_2_name()[self._primary_objective_index]
        primary_objective_variable = momilp_model.objective_name_2_variable()[objective_name]
        constraint_name = "_".join(
            [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, self._num_constraints_for_checked_element])
        momilp_model.add_constraint(primary_objective_variable, constraint_name, objective_value, GRB.EQUAL)
        self._num_constraints_for_checked_element += 1
        # restrict the feasible space to the edge
        left_point = edge.left_point()
        right_point = edge.right_point()
        lambda_var = momilp_model.addVar(name="lambda", ub=1.0)
        # add the convex combination constraint for each criterion
        z1_var = self._projected_space_criterion_index_2_objective_variable(0)
        constraint_name = "_".join(
            [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, self._num_constraints_for_checked_element])
        momilp_model.add_constraint(
            z1_var - lambda_var * (right_point.z1() - left_point.z1()), constraint_name, right_point.z1(), GRB.EQUAL)
        self._num_constraints_for_checked_element += 1
        z2_var = self._projected_space_criterion_index_2_objective_variable(1)
        constraint_name = "_".join(
            [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, self._num_constraints_for_checked_element])
        momilp_model.add_constraint(
            z2_var - lambda_var * (right_point.z2() - left_point.z2()), constraint_name, right_point.z2(), GRB.EQUAL)
        self._num_constraints_for_checked_element += 1
        if not isinstance(edge, EdgeInTwoDimension):
            # then, we should also consider the third objective
            z3_var = primary_objective_variable
            constraint_name = "_".join(
                [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, self._num_constraints_for_checked_element])
            momilp_model.add_constraint(
                z3_var - lambda_var * (right_point.z3() - left_point.z3()), constraint_name, right_point.z3(), 
                GRB.EQUAL)
            self._num_constraints_for_checked_element += 1
        momilp_model.update_model()

    def add_point_constraint(self, point):
        """Adds the constraint to restrict the feasible space to the given point
        
        NOTE: The given point is assumed to be in the original objective space (has the third dimension)"""
        momilp_model = self._momilp_model
        for index, value in enumerate(point.values()):
            objective_name = momilp_model.objective_index_2_name()[index]
            objective_variable = momilp_model.objective_name_2_variable()[objective_name]
            constraint_name = "_".join(
                [DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, self._num_constraints_for_checked_element])
            momilp_model.add_constraint(objective_variable, constraint_name, value, GRB.EQUAL)
            self._num_constraints_for_dominated_space =+ 1
        momilp_model.update_model()

    def fix_integer_vector_of_compared_frontier(self, y_bar):
        """Fixes the current value of the integer vector of the compared frontier in the momilp model"""
        self._momilp_model.fix_integer_vector(y_bar)
        self._momilp_model.update_model()

    def remove_checked_element_constraints(self):
        """Removes the constraints that were added to restrict the feasible space to the element"""
        num_constraints = self._num_constraints_for_checked_element
        constraint_names = [
            "_".join([DominanceModel._CHECKED_ELEMENT_CONSTRAINTS_NAME, i]) for i in range(num_constraints)]
        self._momilp_model.remove_constraint(constraint_names)
        self._num_constraints_for_checked_element = 0

    def remove_dominated_space_constraints(self):
        """Removes the constraints that were added to define the dominated space"""
        num_constraints = self._num_constraints_for_dominated_space
        constraint_names = [
            "_".join([DominanceModel._DOMINATED_SPACE_CONSTRAINTS_NAME, i]) for i in range(num_constraints)]
        self._momilp_model.remove_constraint(constraint_names)
        self._num_constraints_for_dominated_space = 0

    def solve(self, objective_index=None):
        """Solves the model for the objective specified by the objective index, returns the point in the objective 
        space corresponding to the optimal solution
        
        NOTE: A lexicographic objective function is solved where the given objective index is assigned top priority"""
        if objective_index is not None:
            objective_index_2_priority = self._momilp_model.objective_index_2_priority()
            current_priority = objective_index_2_priority[objective_index]
            index_with_top_priority = [
                index for index, priority in objective_index_2_priority.items() if priority == 1][0]
            # swap the priorities
            objective_index_2_priority[objective_index] = 1
            objective_index_2_priority[index_with_top_priority] = current_priority
        # solve
        self._momilp_model.solve()
        values, status = molp_query_utilities.query_optimal_objective_values(
            self._momilp_model.problem(), raise_error_if_infeasible=False)
        return Point(values) if status != GRB.INFEASIBLE else None


class DominanceRules:

    """Implements rules to identify the dominance relations between two sets of points, edges or frontiers"""

    # MOMILP_TO_DO: Dominance rules for open or half-open edges
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


class ModelBasedDominanceFilter:

    """Filters the points that are relatively nondominated with respect to the points or edges checked against"""

    def __init__(self, dominance_model):
        self._dominance_model = dominance_model

    def _solve_model(self, criterion_index):
        """Maximizes the selected criterion in the model, and returns the point corresponding to the optimal solution"""
        return self._dominance_model.solve(criterion_index)

    def _reset_model(self):
        """Removes the constraints related to a previous dominance check if there exist any"""
        self._dominance_model.remove_dominated_space_constraints()
        self._dominance_model.remove_checked_element_constraints()

    def filter_edge(self, edge):
        """Filters the dominated points in the edge, and returns the updated edge"""
        self._dominance_model.add_edge_constraint(edge)
        # solve for z1
        # solve for z2
        self._dominance_model.remove_checked_element_constraints()

    def filter_point(self, point):
        """Returns the point if it is not dominated, otherwise None"""
        self._dominance_model.add_point_constraint(point)
        return self._dominance_model.solve()

    def set_dominated_space(self, frontier_solution, primary_objective_value, reset=True):
        """Sets the dominated space in the model"""
        if reset:
            self._reset_model()
        dominance_model = self._dominance_model
        y_bar = frontier_solution.y_bar()
        dominance_model.fix_integer_vector_of_compared_frontier(y_bar)
        frontier = frontier_solution.frontier()
        dominance_model.add_dominated_space_constraints(frontier, primary_objective_value)
