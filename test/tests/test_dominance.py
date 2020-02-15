"""Implements tests for dominance"""

from hamcrest import assert_that, has_length, is_
import os
from src.common.elements import Edge, EdgeInTwoDimension, FrontierEdgeInTwoDimension, FrontierInTwoDimension, Point, \
    PointInTwoDimension
from src.momilp.dominance import DominanceRules, ModelBasedDominanceFilter
from src.momilp.model import GurobiMomilpModel
from unittest import main, TestCase


class DominanceRulesTest(TestCase):

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


class ModelBasedDominanceFilterTest(TestCase):

    """Implements tests for the model-based dominance filter test"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data")
        self._filter = ModelBasedDominanceFilter(2)
        self.assert_that = assert_that

    def test_edge_dominated_by_edge(self):
        """Tests a scenario where an edge is dominated by a point"""
        first_edge = EdgeInTwoDimension(PointInTwoDimension([1, 5]), PointInTwoDimension([5, 1]))
        second_edge = FrontierEdgeInTwoDimension(PointInTwoDimension([2, 6]), PointInTwoDimension([6, 2]))
        frontier = FrontierInTwoDimension(edges=[second_edge])
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(first_edge)
        self.assert_that(filtered_edges, has_length(0))

    def test_edge_dominated_by_point(self):
        """Tests a scenario where an edge is dominated by a point"""
        edge = EdgeInTwoDimension(PointInTwoDimension([1, 5]), PointInTwoDimension([5, 1]))
        point = PointInTwoDimension([6, 6])
        frontier = FrontierInTwoDimension(point=point)
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(edge)
        self.assert_that(filtered_edges, has_length(0))

    def test_edge_nondominated_relative_to_edge(self):
        """Tests a scenario where an edge is nondominated relative to an edge"""
        first_edge = EdgeInTwoDimension(PointInTwoDimension([2, 4]), PointInTwoDimension([4, 2]))
        second_edge = FrontierEdgeInTwoDimension(PointInTwoDimension([5, 1]), PointInTwoDimension([6, 0]))
        frontier = FrontierInTwoDimension(edges=[second_edge])
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(first_edge)
        self.assert_that(filtered_edges, has_length(1))
        filtered_edge = filtered_edges[0]
        self.assert_that(filtered_edge.left_point(), is_(first_edge.left_point()))
        self.assert_that(filtered_edge.right_point(), is_(first_edge.right_point()))

    def test_edge_nondominated_relative_to_point(self):
        """Tests a scenario where an edge is nondominated relative to a point"""
        edge = EdgeInTwoDimension(PointInTwoDimension([3, 5]), PointInTwoDimension([5, 3]))
        point = PointInTwoDimension([6, 2])
        frontier = FrontierInTwoDimension(point=point)
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(edge)
        self.assert_that(filtered_edges, has_length(1))
        filtered_edge = filtered_edges[0]
        self.assert_that(filtered_edge.left_point(), is_(edge.left_point()))
        self.assert_that(filtered_edge.right_point(), is_(edge.right_point()))

    def test_edge_partially_nondominated_relative_to_edge(self):
        """Tests a scenario where an edge is partially nondominated relative to an edge"""
        first_edge = EdgeInTwoDimension(PointInTwoDimension([1, 4]), PointInTwoDimension([5, 1]))
        second_edge = FrontierEdgeInTwoDimension(PointInTwoDimension([3, 4]), PointInTwoDimension([4, 3]))
        frontier = FrontierInTwoDimension(edges=[second_edge])
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(first_edge)
        self.assert_that(filtered_edges, has_length(1))
        self.assert_that(filtered_edges[0].left_point(), is_(PointInTwoDimension([4, 1.75])))
        self.assert_that(filtered_edges[0].right_point(), is_(PointInTwoDimension([5, 1])))
        self.assert_that(filtered_edges[0].left_inclusive(), is_(False))

    def test_edge_partially_nondominated_relative_to_point(self):
        """Tests a scenario where an edge is partially nondominated relative to a point"""
        edge = EdgeInTwoDimension(PointInTwoDimension([1, 5]), PointInTwoDimension([5, 1]))
        point = PointInTwoDimension([4, 4])
        frontier = FrontierInTwoDimension(point=point)
        self._filter.set_dominated_space(frontier)
        filtered_edges = self._filter.filter_edge(edge)
        self.assert_that(filtered_edges, has_length(2))
        self.assert_that(filtered_edges[0].left_point(), is_(PointInTwoDimension([1,5])))
        self.assert_that(filtered_edges[0].right_point(), is_(PointInTwoDimension([2,4])))
        self.assert_that(filtered_edges[0].right_inclusive(), is_(False))
        self.assert_that(filtered_edges[1].left_point(), is_(PointInTwoDimension([4,2])))
        self.assert_that(filtered_edges[1].right_point(), is_(PointInTwoDimension([5,1])))
        self.assert_that(filtered_edges[1].left_inclusive(), is_(False))

    def test_point_dominated_by_edge(self):
        """Tests a scenario where a point is dominated by an edge"""
        edge = FrontierEdgeInTwoDimension(PointInTwoDimension([1, 5]), PointInTwoDimension([5, 1]))
        point = PointInTwoDimension([2, 2])
        frontier = FrontierInTwoDimension(edges=[edge])
        self._filter.set_dominated_space(frontier)
        nondominated_point = self._filter.filter_point(point)
        self.assert_that(nondominated_point, is_(None))

    def test_point_nondominated_relative_to_edge(self):
        """Tests a scenario where a point is nondominated relative to an edge"""
        edge = FrontierEdgeInTwoDimension(PointInTwoDimension([1, 5]), PointInTwoDimension([5, 1]))
        point = PointInTwoDimension([4, 4])
        frontier = FrontierInTwoDimension(edges=[edge])
        self._filter.set_dominated_space(frontier)
        nondominated_point = self._filter.filter_point(point)
        self.assert_that(nondominated_point, is_(PointInTwoDimension([4, 4])))


if __name__ == '__main__':
    main()
