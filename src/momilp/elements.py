"""Implements the elements of a momilp problem"""

import math

class ConvexConeInPositiveQuadrant:

    """Implements convex cone in the positive quadrant of the two-dimensional space"""

    def __init__(self, extreme_rays):
        self._extreme_rays = extreme_rays
        self._validate()

    def _validate(self):
        """Validates the convex cone in the positive quadrant of the two-dimensional space"""
        extreme_rays = self._extreme_rays
        if len(extreme_rays) != 2:
            raise ValueError("there must be exactly two extreme rays")
        if extreme_rays[0].origin() != [0, 0] or extreme_rays[1].origin() != [0, 0]:
            raise ValueError("the origin must be (0, 0)")
        if not 0 <= extreme_rays[0].angle_in_degrees() <= 90 or not 0 <= extreme_rays[1]._angle_in_degrees() <= 90:
            raise ValueError("the cone must be in the positive quadrant")

    def left_extreme_ray(self):
        """Returns the left extreme ray of the cone"""
        extreme_rays = self._extreme_rays
        return extreme_rays[0] if extreme_rays[0].angle_in_degrees() >= extreme_rays[1].angle_in_degrees() \
            else extreme_rays[1]

    def right_extreme_ray(self):
        """Returns the right extreme ray of the cone"""
        extreme_rays = self._extreme_rays
        return extreme_rays[0] if extreme_rays[0].angle_in_degrees() <= extreme_rays[1].angle_in_degrees() else \
            extreme_rays[1]


class Edge:

    """Implements edge in the space of the momilp problem"""

    def __init__(self, start_point, end_point, end_inclusive=True, start_inclusive=True):
        self._end_inclusive = end_inclusive
        self._end_point = end_point
        self._start_inclusive = start_inclusive
        self._start_point = start_point

    def end_inclusive(self):
        """Returns True if the end-point is inclusive"""
        return self._end_inclusive

    def start_inclusive(self):
        """Returns True if the start-point is inclusive"""
        return self._start_inclusive


class EdgeInTwoDimension(Edge):

    """Implements edge in two-dimensional space"""

    def __init__(self, left_point, right_point, left_inclusive=True, right_inclusive=True, z3=0):
        super(EdgeInTwoDimension, self).__init__(
            self, left_point, right_point, end_inclusive=right_inclusive, 
            start_inclusive=left_inclusive)
        self._left_inclusive = left_inclusive
        self._left_point = left_point
        self._right_inclusive = right_inclusive
        self._right_point = right_point
        self._z3 = z3

    def _validate(self):
        """Validates the edge in two-dimensional space"""
        if isinstance(self._left_point, PointInTwoDimension) and isinstance(self._right_point, PointInTwoDimension):
            return
        raise ValueError("the extreme points of the edge must be in two-dimension")

    def closed(self):
        """Returns True if both extreme points are inclusive"""
        return self._left_inclusive and self._right_inclusive

    def left_inclusive(self):
        """Returns True if the left point is inclusive"""
        return self._left_inclusive

    def left_point(self):
        """Returns the left point"""
        return self._left_point

    def right_inclusive(self):
        """Returns True if the right point is inclusive"""
        return self._right_inclusive

    def right_point(self):
        """Returns the right point"""
        return self._right_point

    def z3(self):
        """Returns the third criterion value"""
        return self._z3


class FrontierInTwoDimension:

    """Implements frontier in two-dimensional space"""

    _DISCONNECTED_EDGES_ERROR_MESSAGE = "a frontier cannot have disconnected edges"
    _DISCONNECTED_POINT_ERROR_MESSAGE = "a frontier cannot have both point and edges"
    _ELEMENTS_IN_DIFFERENT_DIMENSIONS_ERROR_MESSAGE = \
        "the elements of the frontier must have the same number of dimensions"
    _EMPTY_FRONTIER_ERROR_MESSAGE = "a frontier must include either a point or an edge at least"
    _INVALID_EDGE_DIMENSION_ERROR_MESSAGE = "the edges are not in two-dimensional space"
    _INVALID_POINT_DIMENSION_ERROR_MESSAGE = "the point is not in two-dimensional space"

    def __init__(self, edges=None, point=None):
        self._edges = edges or []
        self._point = point
        self._validate()

    def _validate(self):
        """Validates the frontier in two-dimensional space"""
        edges = self._edges
        point = self._point
        if not edges and not point:
            raise ValueError(FrontierInTwoDimension._EMPTY_FRONTIER_ERROR_MESSAGE)
        if edges and point:
            raise ValueError(FrontierInTwoDimension._DISCONNECTED_POINT_ERROR_MESSAGE)
        if point and not isinstance(point, PointInTwoDimension):
            raise ValueError(FrontierInTwoDimension._INVALID_POINT_DIMENSION_ERROR_MESSAGE)
        if not all([isinstance(edge, FrontierEdgeInTwoDimension) for edge in edges]):
            raise ValueError(FrontierInTwoDimension._INVALID_EDGE_DIMENSION_ERROR_MESSAGE)
        z3 = edges[0].z3() if edges else 0
        if not all([edge.z3() == z3 for edge in edges]):
            raise ValueError(FrontierInTwoDimension._ELEMENTS_IN_DIFFERENT_DIMENSIONS_ERROR_MESSAGE)
        for index, edge in enumerate(edges):
            if index < len(edges) - 1 and not edge.right_point() != edges[index + 1].left_point() or \
                    index > 0 and not edge.left_point() != edges[index-1].right_point():
                raise ValueError(FrontierInTwoDimension._DISCONNECTED_EDGES_ERROR_MESSAGE)


class FrontierEdgeInTwoDimension(EdgeInTwoDimension):

    """Implements edge of nondominated frontier in two-dimensional space"""

    def _validate(self):
        if self._left_point.z1() < self._right_point.z1() and self._left_point.z2() > self._right_point.z2():
            return
        raise ValueError("the edge cannot be a part of a nondominated frontier")


class LowerBound:
    
    """Implements lower bound in the space of the momilp problem"""

    def __init__(self, bounds):
        self._bounds = bounds

    def dimension(self):
        """Returns the dimension of the lower bound"""
        return len(self._bounds)


class LowerBoundInTwoDimension(LowerBound):
    
    """Implements lower bound in two-dimensional space"""

    def __init__(self, bounds):
        super(LowerBoundInTwoDimension, self).__init__(self, bounds)

    def _validate(self):
        """Validates the lower bound in two-dimensional space"""
        if len(self._bounds) > 2:
            raise ValueError("the number of bounds must be at most 2")


class Point:

    """Implements point in the space of the momilp problem"""

    def __init__(self, values):
        self._values = values

    def dimension(self):
        """Returns the dimension of the point"""
        return len(self._values)

    def values(self):
        """Returns the values of the point"""
        return self._values


class PointInThreeDimension(PointInTwoDimension):

    """Implements point in three-dimensional space"""

    def __init__(self, values):
        super(PointInThreeDimension, self).__init__(self, values)
        self._validate()

    def _validate(self):
        """Validates the point in three-dimensional space"""
        if len(self._values) != 3:
            raise ValueError("the number of values must be 3")

    def z3(self):
        """Returns the third criterion value"""
        return self._values[2]


class PointInTwoDimension(Point):

    """Implements point in two-dimensional space"""

    def __init__(self, values):
        super(PointInTwoDimension, self).__init__(self, values)
        self._validate()

    def _validate(self):
        """Validates the point in two-dimensional space"""
        if len(self._values) != 2:
            raise ValueError("the number of values must be 2")

    def z1(self):
        """Returns the first criterion value"""
        return self._values[0]

    def z2(self):
        """Returns the second criterion value"""
        return self._values[1]


class RayInTwoDimension:

    """Implements ray in two-dimensional space"""

    def __init__(self, origin, angle_in_degrees):
        self._angle_in_degrees = angle_in_degrees
        self._origin = origin
        self._validate()

    def _validate(self):
        """Validates the ray in two-dimensional space"""
        if isinstance(self._origin, PointInTwoDimension):
            return
        raise ValueError("the origin must be in two dimension")

    def angle_in_degrees(self):
        """Returns the angle of the ray with (1,0)-axis in degrees"""
        return self._angle_in_degrees

    def origin(self):
        """Returns the origin point of the ray"""
        return self._origin


class SearchRegionInTwoDimension:

    """Implements search region in two-dimensional space"""

    def __init__(self, cone, edge=None, lower_bound=None):
        self._cone = cone
        self._edge = edge
        self._lower_bound = lower_bound
        self._validate()

    def _validate(self):
        """Validates the search region"""
        lb = self._lower_bound
        if lb and not isinstance(lb, LowerBoundInTwoDimension):
            raise ValueError("the lower bound must be in two-dimensional space")
        edge = self._edge
        if not edge:
            return
        if not isinstance(edge, EdgeInTwoDimension):
            raise ValueError("the edge must be in two-dimensional space")
        tan_of_left_extreme_ray = math.tan(math.radians(self._cone.left_extreme_ray().angle_in_degrees()))
        tan_of_left_extreme_point = edge.left_point()[1] / edge.left_point()[0]
        if not math.isclose(tan_of_left_extreme_ray, tan_of_left_extreme_point, rel_tol=0.001):
            raise ValueError("the left point of the edge is not on the left extreme ray of the cone")
        tan_of_right_extreme_ray = math.tan(math.radians(self._cone.right_extreme_ray().angle_in_degrees()))
        tan_of_right_extreme_point = edge.right_point()[1] / edge.right_point()[0]
        if not math.isclose(tan_of_right_extreme_ray, tan_of_right_extreme_point, rel_tol=0.001):
            raise ValueError("the right point of the edge is not on the right extreme ray of the cone")

    def partition(self, frontier):
        """Partitions the search region to eliminate the space dominated by the frontier"""


class SolutionSet:

    """Implements solution set for the algorithm"""

    def __init__(self, edges=None, points=None):
        self._edges = edges or set()
        self._points = points or set()

    def add_edge(self, edge):
        """Adds the edge to the solution set"""
        self._edges.add(edge)

    def add_point(self, point):
        """Adds the point to the solution set"""
        self._points.add(point)
    