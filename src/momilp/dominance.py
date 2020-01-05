"""Checks the dominance of a set of points, and eliminates the dominated points"""

import numpy as np
from src.common.elements import PointInTwoDimension, EdgeInTwoDimension, FrontierInTwoDimension

class DominanceModel:

    """Implements model to check the dominance of a point or an edge"""


class DominanceRules:

    """Implements rules to identify the dominance relations between two sets of points, edges or frontiers"""

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
    
    class PointToPoint:

        """Implements the rules when comparing a point to another point"""

        @staticmethod
        def dominated(this, that):
            """Returns True if 'this' point is dominated by 'that' point, otherwise False"""
            assert len(this.values()) == len(that.values()), \
                "the compared points must have the same number of dimensions"
            return this != that and all([v <= that.values()[i] for i, v in enumerate(this.values())])

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
            if point.z2() > frontier.edges()[0].left_point().z2():
                return False
            if point.z1() > frontier.edges()[-1].right_point().z1():
                return False
            for edge in frontier.edges():
                if np.dot(point.values(), edge.normal_vector()) >= edge.edge_value():
                    return False
            return True


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