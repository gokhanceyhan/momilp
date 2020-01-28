"""Implements tests for dominance"""

from hamcrest import assert_that
from src.common.elements import Edge, EdgeInTwoDimension, FrontierEdgeInTwoDimension, FrontierInTwoDimension, Point, \
    PointInTwoDimension
from src.momilp.dominance import DominanceRules
from unittest import main, TestCase


class TestDominanceRules(TestCase):

    """Implements tests for the dominance rules"""

    def setUp(self):
        self.assert_that = assert_that

    def test_edge_dominated_by_edge_in_two_dimension(self):
        """Tests a scenario where an edge is dominated by another edge in two dimension"""
        this = EdgeInTwoDimension(PointInTwoDimension([3, 6]), PointInTwoDimension([5, 5]))
        that = EdgeInTwoDimension(PointInTwoDimension([3, 8]), PointInTwoDimension([5, 6]))
        self.assert_that(DominanceRules.EdgeToEdge.dominated(this, that))

    def test_edge_partially_nondominated_relative_to_edge_in_two_dimension(self):
        """Tests a scenario where an edge is partially nondominated compared to another edge in two dimension"""
        this = EdgeInTwoDimension(PointInTwoDimension([5, 5]), PointInTwoDimension([3, 8]))
        that = EdgeInTwoDimension(PointInTwoDimension([5, 5]), PointInTwoDimension([4, 9]))
        self.assert_that(not DominanceRules.EdgeToEdge.dominated(this, that))

    def test_edge_dominated_by_frontier_in_two_dimension(self):
        """Tests a scenario where an edge is dominated by a frontier in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([11,11]))
        self.assert_that(DominanceRules.EdgeToFrontier.dominated(edge, frontier))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([6, 10]), PointInTwoDimension([8, 9])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([8, 9]), PointInTwoDimension([10, 6]))])
        self.assert_that(DominanceRules.EdgeToFrontier.dominated(edge, frontier))

    def test_edge_partially_nondominated_relative_to_frontier_in_two_dimension(self):
        """Tests a scenario where an edge is partially nondominated compared to a frontier in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([7, 7]))
        self.assert_that(not DominanceRules.EdgeToFrontier.dominated(edge, frontier))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([11, 5]))])
        self.assert_that(not DominanceRules.EdgeToFrontier.dominated(edge, frontier))

    def test_edge_dominated_by_point(self):
        """Tests a scenario where an edge is dominated by a point"""
        point = Point([10, 10, 10])
        edge = Edge(Point([5, 5, 5]), Point([8, 8, 8]))
        self.assert_that(DominanceRules.EdgeToPoint.dominated(edge, point))
        edge = Edge(Point([5, 10, 10]), Point([10, 5, 10]))
        self.assert_that(DominanceRules.EdgeToPoint.dominated(edge, point))

    def test_edge_partially_nondominated_relative_to_point(self):
        """Tests a scenario where an edge is partially nondominated compared to a point"""
        point = Point([10, 10])
        edge = Edge(Point([5, 11]), Point([8, 8]))
        self.assert_that(not DominanceRules.EdgeToPoint.dominated(edge, point))

    def test_frontier_dominated_by_edge_in_two_dimension(self):
        """Tests a scenario where a frontier is dominated by an edge in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([6, 6]))
        self.assert_that(DominanceRules.FrontierToEdge.dominated(frontier, edge))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([5, 5]))])
        self.assert_that(DominanceRules.FrontierToEdge.dominated(frontier, edge))

    def test_frontier_partially_nondominated_relative_to_edge_in_two_dimension(self):
        """Tests a scenario where a frontier is partially nondominated compared to an edge in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([8, 8]))
        self.assert_that(not DominanceRules.FrontierToEdge.dominated(frontier, edge))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([11, 5]))])
        self.assert_that(not DominanceRules.FrontierToEdge.dominated(frontier, edge))

    def test_frontier_dominated_by_frontier_in_two_dimension(self):
        """Tests a scenario where a frontier is dominated by another frontier in two dimension"""
        this = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([5, 5]))])
        that = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([2, 9]), PointInTwoDimension([3, 8])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 8]), PointInTwoDimension([5, 6]))])
        self.assert_that(DominanceRules.FrontierToFrontier.dominated(this, that))

    def test_frontier_partially_nondominated_relative_to_frontier_in_two_dimension(self):
        """Tests a scenario where a frontier is partially nondominated compared to another frontier in two dimension"""
        this = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([5, 5]))])
        that = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([2, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([5, 6]))])
        self.assert_that(not DominanceRules.FrontierToFrontier.dominated(this, that))

    def test_frontier_dominated_by_point_in_two_dimension(self):
        """Tests a scenario where a frontier is dominated by a point in two dimension"""
        point = Point([10, 10])
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([8, 8]))
        self.assert_that(DominanceRules.FrontierToPoint.dominated(frontier, point))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([5, 5]))])
        self.assert_that(DominanceRules.FrontierToPoint.dominated(frontier, point))

    def test_frontier_partially_nondominated_relative_to_point_in_two_dimension(self):
        """Tests a scenario where a frontier is partially nondominated compared to a point in two dimension"""
        point = Point([10, 10])
        frontier = FrontierInTwoDimension(point=PointInTwoDimension([10, 10]))
        self.assert_that(not DominanceRules.FrontierToPoint.dominated(frontier, point))
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([1, 9]), PointInTwoDimension([3, 7])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([3, 7]), PointInTwoDimension([11, 5]))])
        self.assert_that(not DominanceRules.FrontierToPoint.dominated(frontier, point))

    def test_point_dominated_by_edge_in_two_dimension(self):
        """Tests a scenario where a point is dominated by an edge in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        point = PointInTwoDimension([5, 5])
        self.assert_that(DominanceRules.PointToEdge.dominated(point, edge))
        point = PointInTwoDimension([4, 10])
        self.assert_that(DominanceRules.PointToEdge.dominated(point, edge))
        point = PointInTwoDimension([6, 6])
        self.assert_that(DominanceRules.PointToEdge.dominated(point, edge))

    def test_point_nondominated_relative_to_edge_in_two_dimension(self):
        """Tests a scenario where a point is relatively nondominated compared to an edge in two dimension"""
        edge = EdgeInTwoDimension(PointInTwoDimension([5, 10]), PointInTwoDimension([10, 5]))
        point = PointInTwoDimension([5, 11])
        self.assert_that(not DominanceRules.PointToEdge.dominated(point, edge))
        point = PointInTwoDimension([11, 10])
        self.assert_that(not DominanceRules.PointToEdge.dominated(point, edge))
        point = PointInTwoDimension([8, 8])
        self.assert_that(not DominanceRules.PointToEdge.dominated(point, edge))

    def test_point_dominated_by_frontier_in_two_dimension(self):
        """Tests a scenario where a point is dominated by a frontier in two dimension"""
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([2, 10]), PointInTwoDimension([6, 8])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([6, 8]), PointInTwoDimension([8, 4]))])
        point = PointInTwoDimension([5, 5])
        self.assert_that(DominanceRules.PointToFrontier.dominated(point, frontier))
        point = PointInTwoDimension([4, 8])
        self.assert_that(DominanceRules.PointToFrontier.dominated(point, frontier))
        point = PointInTwoDimension([7, 5])
        self.assert_that(DominanceRules.PointToFrontier.dominated(point, frontier))

    def test_point_nondominated_relative_to_frontier_in_two_dimension(self):
        """Tests a scenario where a point is relatively nondominated compared to a frontier in two dimension"""
        frontier = FrontierInTwoDimension(
            edges=[
                FrontierEdgeInTwoDimension(PointInTwoDimension([2, 10]), PointInTwoDimension([6, 8])), 
                FrontierEdgeInTwoDimension(PointInTwoDimension([6, 8]), PointInTwoDimension([8, 4]))])
        point = PointInTwoDimension([5, 9])
        self.assert_that(not DominanceRules.PointToFrontier.dominated(point, frontier))
        point = PointInTwoDimension([7, 7])
        self.assert_that(not DominanceRules.PointToFrontier.dominated(point, frontier))
        point = PointInTwoDimension([9, 5])
        self.assert_that(not DominanceRules.PointToFrontier.dominated(point, frontier))

    def test_point_dominated_by_point(self):
        """Tests a scenario where a point is dominated by another point"""
        base_point = Point([10, 10, 10])
        compared_point = Point([10, 11, 10])
        self.assert_that(DominanceRules.PointToPoint.dominated(base_point, compared_point))

    def test_point_nondominated_relative_to_point(self):
        """Tests a scenario where a point is relatively nondominated compared to another point"""
        base_point = Point([10, 10, 10])
        compared_point = Point([10, 10, 10])
        self.assert_that(not DominanceRules.PointToPoint.dominated(base_point, compared_point))
        compared_point = Point([11, 10, 9])
        self.assert_that(not DominanceRules.PointToPoint.dominated(base_point, compared_point))


if __name__ == '__main__':
    main()