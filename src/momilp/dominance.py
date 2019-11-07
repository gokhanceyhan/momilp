"""Checks the dominance of a set of points, and eliminates the dominated points"""

class DominanceModel:

    """Implements model to check the dominance of a point or an edge"""


class DominanceFilter:

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


class DominanceRuleFactory:

    """Implements rules to identify the dominance relations between two sets of points"""

    class EdgeToPoint:

        """Implements the rules when comparing an edge to a point"""

    class FrontierToPoint:

        """Implements the rules when comparing a frontier to a point"""
    
    class PointToPoint:

        """Implements the rules when comparing a point to another point"""

    class PointToEdge:

        """Implements the rules when comparing a point to an edge"""

    class PointToFrontier:

        """Implements the rules when comparing a point to frontier"""
