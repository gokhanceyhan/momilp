"""Implements the elements of a momilp problem"""

import abc
from enum import Enum
import math
import numpy as np


def point_on_ray_in_two_dimension(point, ray, tol=1e-4):
    """Returns True of the point is on the ray with the specified error tolerance, False otherwise"""
    tan_of_ray = math.tan(math.radians(ray.angle_in_degrees())) if ray.angle_in_degrees() < 90 else float("inf")
    tan_of_point = point.z2() / point.z1() if point.z1() else float("inf")
    return math.isclose(tan_of_ray, tan_of_point, rel_tol=tol)


class ConvexConeInPositiveQuadrant:

    """Implements convex cone in the positive quadrant of the two-dimensional space"""

    def __init__(self, extreme_rays):
        self._extreme_rays = extreme_rays
        self._validate()

    def __str__(self):
        return "Cone: {%s, %s}" % (self._extreme_rays[0], self._extreme_rays[1])

    def _validate(self):
        """Validates the convex cone in the positive quadrant of the two-dimensional space"""
        extreme_rays = self._extreme_rays
        if len(extreme_rays) != 2:
            raise ValueError("there must be exactly two extreme rays")
        if extreme_rays[0].origin().values() != [0, 0] or extreme_rays[1].origin().values() != [0, 0]:
            raise ValueError("the origin must be (0, 0)")
        if not 0 <= extreme_rays[0].angle_in_degrees() <= 90 or not 0 <= extreme_rays[1].angle_in_degrees() <= 90:
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

    def __str__(self):
        prefix = "[" if self._start_inclusive else "("
        suffix = "]" if self._end_inclusive else ")"
        return "Edge: " + prefix + "%s, %s" % (self._start_point, self._end_point) + suffix

    def end_inclusive(self):
        """Returns True if the end-point is inclusive"""
        return self._end_inclusive

    def end_point(self):
        """Returns the end point of the edge"""
        return self._end_point

    def start_inclusive(self):
        """Returns True if the start-point is inclusive"""
        return self._start_inclusive

    def start_point(self):
        """Returns the start point of the edge"""
        return self._start_point


class EdgeInTwoDimension(Edge):

    """Implements edge in two-dimensional space"""

    def __init__(self, left_point, right_point, left_inclusive=True, right_inclusive=True, z3=0):
        super(EdgeInTwoDimension, self).__init__(
            left_point, right_point, end_inclusive=right_inclusive, start_inclusive=left_inclusive)
        self._edge_value = None
        self._left_inclusive = left_inclusive
        self._left_point = left_point
        self._normal_vector = None
        self._right_inclusive = right_inclusive
        self._right_point = right_point
        self._z3 = z3
        self._validate()
        self._set_normal_vector()
        self._set_edge_value()

    def _set_edge_value(self):
        """Sets the edge value"""
        self._edge_value = np.dot(self._normal_vector, self._left_point.values())

    def _set_normal_vector(self):
        """Sets the normal vector of the edge"""
        self._normal_vector = [
            self._left_point.z2() - self._right_point.z2(), self._right_point.z1() - self._left_point.z1()]

    def _validate(self):
        """Validates the edge in two-dimensional space"""
        if isinstance(self._left_point, PointInTwoDimension) and isinstance(self._right_point, PointInTwoDimension):
            return
        raise ValueError("the extreme points of the edge must be in two-dimension")

    def closed(self):
        """Returns True if both extreme points are inclusive"""
        return self._left_inclusive and self._right_inclusive

    def edge_value(self):
        """Returns the edge value"""
        return self._edge_value

    def left_inclusive(self):
        """Returns True if the left point is inclusive"""
        return self._left_inclusive

    def left_point(self):
        """Returns the left point"""
        return self._left_point

    def normal_vector(self):
        """Returns the normal vector of the edge"""
        return self._normal_vector

    def right_inclusive(self):
        """Returns True if the right point is inclusive"""
        return self._right_inclusive

    def right_point(self):
        """Returns the right point"""
        return self._right_point

    def z3(self):
        """Returns the third criterion value"""
        return self._z3


class Solution:

    """Implements partial solution to the problem"""

    def __init__(self, y_bar):
        self._y_bar = y_bar

    def y_bar(self):
        """Returns the y-vector corresponding to the solution"""
        return self._y_bar

    
class EdgeSolution(Solution):

    """Implements a solution to the problem with a corresponding edge in the objective space"""

    def __init__(self, edge, y_bar):
        super(EdgeSolution, self).__init__(y_bar)
        self._edge = edge

    def edge(self):
        """Returns the edge"""
        return self._edge


class FrontierEdgeInTwoDimension(EdgeInTwoDimension):

    """Implements edge of nondominated frontier in two-dimensional space"""

    def _validate(self):
        if self._left_point.z1() < self._right_point.z1() and self._left_point.z2() > self._right_point.z2():
            return
        raise ValueError("the edge cannot be a part of a nondominated frontier")


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

    def __str__(self):
        return "Frontier: " + (("edges= {" + ", ".join([str(e) for e in self._edges]) + "}") if self._edges else \
            "point=%s" % self._point)

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
            if index < len(edges) - 1 and edge.right_point() != edges[index + 1].left_point() or \
                    index > 0 and edge.left_point() != edges[index-1].right_point():
                raise ValueError(FrontierInTwoDimension._DISCONNECTED_EDGES_ERROR_MESSAGE)

    def edges(self):
        """Returns the edges in the frontier (sorted in non-increasing values of z1 (x-axis), from left to right)"""
        return self._edges

    def point(self):
        """Returns the point in the frontier"""
        return self._point

    def singleton(self):
        """Returns True if the frontier is a singleton, otherwise False"""
        return self._point and not self._edges 


class FrontierSolution(Solution):

    """Implements a solution to the problem with a corresponding frontier in the objective space"""

    def __init__(self, frontier, y_bar):
        super(FrontierSolution, self).__init__(y_bar)
        self._frontier = frontier

    def frontier(self):
        """Returns the frontier"""
        return self._frontier

    def y_bar(self):
        """Returns the integer vector"""
        return self._y_bar


class LowerBound:
    
    """Implements lower bound in the space of the momilp problem"""

    def __init__(self, bounds):
        self._bounds = bounds

    def __str__(self):
        return "LB: [" + ", ".join([str(b) for b in self._bounds]) + "]"

    def bounds(self):
        """Returns the bounds"""
        return self._bounds

    def dimension(self):
        """Returns the dimension of the lower bound"""
        return len(self._bounds)


class LowerBoundInTwoDimension(LowerBound):
    
    """Implements lower bound in two-dimensional space"""

    def __init__(self, bounds):
        super(LowerBoundInTwoDimension, self).__init__(bounds)

    def _validate(self):
        """Validates the lower bound in two-dimensional space"""
        if len(self._bounds) > 2:
            raise ValueError("the number of bounds must be at most 2")

    def z1(self):
        """Returns the lower bound on the first criterion value"""
        return self._bounds[0]

    def z2(self):
        """Returns the lower bound on the second criterion value"""
        return self._bounds[1] if len(self._bounds) > 1 else None


class OptimizationStatus(Enum):

    """Represents an optimization status"""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNDEFINED = "undefined"


class Point:

    """Implements point in the space of the momilp problem"""

    def __eq__(self, other):
        if isinstance(other, Point):
            return self._values == other.values()
        return False

    def __init__(self, values):
        self._values = values

    def __str__(self):
        return "Point: (" + ", ".join([str(v) for v in self._values]) + ")"

    def dimension(self):
        """Returns the dimension of the point"""
        return len(self._values)

    def values(self):
        """Returns the values of the point"""
        return self._values


class PointInTwoDimension(Point):

    """Implements point in two-dimensional space"""

    def __init__(self, values):
        super(PointInTwoDimension, self).__init__(values)
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


class PointInThreeDimension(PointInTwoDimension):

    """Implements point in three-dimensional space"""

    def __init__(self, values):
        super(PointInThreeDimension, self).__init__(values)
        self._validate()

    def _validate(self):
        """Validates the point in three-dimensional space"""
        if len(self._values) != 3:
            raise ValueError("the number of values must be 3")

    def z3(self):
        """Returns the third criterion value"""
        return self._values[2]


class PointSolution(Solution):

    """Implements a solution to the problem with a corresponding point in the objective space"""

    def __init__(self, point, y_bar):
        super(PointSolution, self).__init__(y_bar)
        self._point = point

    def __str__(self):
        return "Point Solution: {%s, y=%s}" % (self._point, self._y_bar)

    def point(self):
        """Returns the point"""
        return self._point


class RayInTwoDimension:

    """Implements ray in two-dimensional space"""

    def __init__(self, angle_in_degrees, origin):
        self._angle_in_degrees = angle_in_degrees
        self._origin = origin
        self._validate()

    def __str__(self):
        return "Ray: {origin: %s, angle: %s}" % (self._origin, self._angle_in_degrees)

    def _validate(self):
        """Validates the ray in two-dimensional space"""
        if isinstance(self._origin, PointInTwoDimension):
            return
        raise ValueError("the origin must be a point in two dimension")

    def angle_in_degrees(self):
        """Returns the angle of the ray with (1,0)-axis in degrees"""
        return self._angle_in_degrees

    def origin(self):
        """Returns the origin point of the ray"""
        return self._origin


class SearchProblemResult:

    """Implements search problem result"""

    def __init__(self, point_solution, status):
        self._point_solution = point_solution
        self._status = status

    def point_solution(self):
        """Returns the point solution"""
        return self._point_solution

    def status(self):
        """Returns the optimization status"""
        return self._status
    

class SearchRegion(metaclass=abc.ABCMeta):

    """Implements search region"""

    @abc.abstractmethod
    def dim(self):
        """Returns the dimension of the search region"""


class SearchRegionInTwoDimension(SearchRegion):

    """Implements search region in two-dimensional space"""

    def __init__(self, x_obj_name, y_obj_name, cone, edge=None, lower_bound=None, id_=None):
        self._cone = cone
        self._dim = 2
        self._edge = edge
        self._lower_bound = lower_bound
        self._id = id_ or str(id(self))
        self._x_obj_name = x_obj_name
        self._y_obj_name = y_obj_name
        self._validate()

    def __str__(self):
        elements = [self._cone]
        if self._edge:
            elements.append(self._edge)
        if self._lower_bound:
            elements.append(self._lower_bound)
        return "Region: " + ", ".join([str(e) for e in elements])

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
        left_extreme_ray = self._cone.left_extreme_ray()
        if not point_on_ray_in_two_dimension(edge.left_point(), left_extreme_ray):
            raise ValueError(
                "the left point of the edge '%s' is not on the left extreme ray of the cone with origin '%s' and " \
                "angle in degrees '%s'" % (
                    edge.left_point().values(), left_extreme_ray.origin().values(), 
                    left_extreme_ray.angle_in_degrees()))
        right_extreme_ray = self._cone.right_extreme_ray()
        if not point_on_ray_in_two_dimension(edge.right_point(), right_extreme_ray):
            raise ValueError(
                "the right point of the edge '%s' is not on the right extreme ray of the cone with origin '%s' and " \
                "angle in degrees '%s'" % (
                    edge.right_point().values(), right_extreme_ray.origin().values(), 
                    right_extreme_ray.angle_in_degrees()))

    def cone(self):
        """Returns the cone"""
        return self._cone

    def dim(self):
        return self._dim

    def edge(self):
        """Returns the edge"""
        return self._edge

    def id(self):
        """Returns the region id"""
        return self._id

    def lower_bound(self):
        """Returns the lower bound"""
        return self._lower_bound

    def set_lower_bound(self, lower_bound):
        """Sets the lower bound of the region to the given bound"""
        if not isinstance(lower_bound, LowerBoundInTwoDimension):
            raise ValueError("the lower bound must be in two-dimensional space")
        self._lower_bound = lower_bound

    def x_obj_name(self):
        """Returns the obj name for the x-axis of the region"""
        return self._x_obj_name

    def y_obj_name(self):
        """Returns the obj name for the y-axis of the region"""
        return self._y_obj_name


class SearchStatus(Enum):

    """Implements search status"""

    COMPLETED = 0
    TERMINATED_BY_TIME_LIMIT = 1
    TERMINATED_WITH_ERROR = 2


class SliceProblemResult:

    """Implements slice problem result"""

    def __init__(self, frontier_solution, ideal_point, status=None):
        self._frontier_solution = frontier_solution
        self._ideal_point = ideal_point
        self._status = status

    def frontier_solution(self):
        """Returns the frontier solution"""
        return self._frontier_solution

    def ideal_point(self):
        """Returns the ideal point which is set after the frontier is generated"""
        return self._ideal_point

    def status(self):
        """Returns the optimization status"""
        return self._status


class SolutionSet:

    """Implements solution set for the algorithm"""

    def __init__(self, edge_solutions=None, point_solutions=None):
        self._edge_solutions = edge_solutions or set()
        self._point_solutions = point_solutions or set()

    def add_edge_solution(self, edge_solution):
        """Adds the edge solution to the solution set"""
        self._edge_solutions.add(edge_solution)

    def add_point_solution(self, point_solution):
        """Adds the point solution to the solution set"""
        self._point_solutions.add(point_solution)


class SolverPackage(Enum):

    """Represents a solver package"""

    CPLEX = "cplex"
    GUROBI = "gurobi"


class SolverStage(Enum):

    """Represents the solver stage"""

    MODEL_SCALING = "model scaling"
    SEARCH = "search"
