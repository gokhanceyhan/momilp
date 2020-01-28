"""Checks the dominance of a set of points, and eliminates the dominated points"""

import numpy as np
from src.common.elements import PointInTwoDimension, EdgeInTwoDimension, FrontierInTwoDimension

class DominanceModel:

    """Implements model to check the dominance of a point or an edge"""


class DominanceRules:

    """Implements rules to identify the dominance relations between two sets of points, edges or frontiers"""

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

    def __init__(self, model):
        self._model = model

    def _create_model(self):
        """Creates the model to check the dominance"""
        pass

    def _solve_model(self, criterion_index):
        """Maximizes the selected criterion in the model"""
        pass

    def _update_model(self, y, compared_edges=None, compared_points=None):
        """Updates the model"""
        pass

    def filter_edge(self, y, edge, compared_edges=None, compared_points=None):
        """Filters the dominated points in the edge, and returns the updated edge"""
        pass

    def filter_point(self, y, point, compared_edges=None, compared_points=None):
        """Returns the point if it is not dominated, otherwise None"""
        pass