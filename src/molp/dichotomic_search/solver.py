"""Implements dichotomic search to find extreme supported nondominated points of BOLP problems

Based on:

Aneja, Yash P., and Kunhiraman PK Nair. "Bicriteria transportation problem." Management Science 25.1 (1979): 73-78."""

from collections import namedtuple
import math
from src.common.elements import PointInTwoDimension
from src.molp.solver import MolpSolver
from src.molp.utilities import ModelQueryUtilities


PointPair = namedtuple("PointPair", ["point_with_higher_z1_value", "point_with_higher_z2_value"])


class BolpDichotomicSearchWithGurobiSolver(MolpSolver):

    """Implements bi-objective linear programming problem solver by applying dichotomic search with Gurobi solver"""

    def __init__(self, model, obj_rel_tol=1e-6):
        super(BolpDichotomicSearchWithGurobiSolver, self).__init__(model)
        self._extreme_supported_nondominated_points = []
        self._obj_rel_tol = obj_rel_tol
        self._point_pairs_to_check = []
        self._set_model_params()
        self._validate()
        self._initialize()

    def _calculate_objective_weights(self, point_with_higher_z1_value, point_with_higher_z2_value):
        """Calculates the objective weights to search a nondominated point between the given two points"""
        w_0 = point_with_higher_z2_value.z2() - point_with_higher_z1_value.z2()
        w_1 = point_with_higher_z1_value.z1() - point_with_higher_z2_value.z1()
        w = w_0 + w_1
        return {0: w_0 / w, 1: w_1 / w}
        
    def _initialize(self):
        """Initializes the solver"""
        model = self._model
        # first (second) indexed point in the extreme supported points is the one with the highest z1 (z2) value
        points = []
        for i in range(2):
            model.setParam("ObjNumber", 0)
            model.setAttr("ObjNPriority", 2 - i)
            model.setParam("ObjNumber", 1)
            model.setAttr("ObjNPriority", 1 + i)
            model.optimize()
            values, _ = ModelQueryUtilities.query_optimal_objective_values(model)
            new_point = PointInTwoDimension(values)
            if points and self._isclose(points[-1], new_point):
                continue
            points.append(new_point)
        self._extreme_supported_nondominated_points = points
        if len(points) < 2:
            return
        self._point_pairs_to_check.append(
            PointPair(
                point_with_higher_z1_value=self._extreme_supported_nondominated_points[0], 
                point_with_higher_z2_value=self._extreme_supported_nondominated_points[1]))

    def _isclose(self, point_a, point_b, rel_tol=1e-6):
        """Returns True if two points are close in their values, False otherwise"""
        return math.isclose(point_a.z1(), point_b.z1(), rel_tol=rel_tol) and \
            math.isclose(point_a.z2(), point_b.z2(), rel_tol=rel_tol)

    def _modify_model_objectives(self, equal_priority=False, obj_index_2_weight=None):
        """Modify the model objectives"""
        model = self._model
        if equal_priority:
            for i in range(2):
                model.setParam("ObjNumber", i)
                model.setAttr("ObjNPriority", 1)
        obj_index_2_weight = obj_index_2_weight or {}
        for obj_index, weight in obj_index_2_weight.items():
            model.setParam("ObjNumber", obj_index)
            model.setAttr("ObjNWeight", weight)

    def _set_model_params(self, obj_n_abs_tol=0):
        """Sets the model parameters
        
        NOTE: The value of the 'ObjNAbsTol' parameter indicates the amount by which a fixed variable's reduced cost is 
        allowed to violate dual feasibility, whereas the 'ObjNRelTol' parameter is simply ignored 
        (https://www.gurobi.com/documentation/9.0/refman/working_with_multiple_obje.html)"""
        model = self._model
        for index in range(2):
            model.setParam("ObjNumber", index)
            model.setAttr("ObjNAbsTol", obj_n_abs_tol)
        model.update()

    @staticmethod
    def _sort_points(points, key, reverse=False):
        """Sorts the points"""
        return sorted(points, key=key, reverse=reverse)

    def _validate(self):
        """Validates that the model is a BOLP"""
        model = self._model
        assert model.getAttr("NumObj") == 2, "the'%s' model is not bi-objective" % self._model.getAttr("ModelName")
        assert not model.isQP, "the %s model is a QP" % self._model.getAttr("ModelName")
        assert not model.isQCP, "the %s model is a QCP" % self._model.getAttr("ModelName")

    def extreme_supported_nondominated_points(self):
        """Returns the extreme supported nondominated points"""
        return self._extreme_supported_nondominated_points
    
    def solve(self):
        model = self._model
        point_pairs_two_check = self._point_pairs_to_check
        while len(point_pairs_two_check) > 0:
            point_pair = point_pairs_two_check.pop(0)
            point_with_higher_z1_value = point_pair.point_with_higher_z1_value
            point_with_higher_z2_value = point_pair.point_with_higher_z2_value
            obj_index_2_weight = self._calculate_objective_weights(
                point_with_higher_z1_value, point_with_higher_z2_value)
            self._modify_model_objectives(equal_priority=True, obj_index_2_weight=obj_index_2_weight)
            model.optimize()
            values, _ = ModelQueryUtilities.query_optimal_objective_values(model)
            point = PointInTwoDimension(values)
            if self._isclose(point, point_with_higher_z1_value) or self._isclose(point, point_with_higher_z2_value):
                continue
            left_extreme_point_obj_value = point_with_higher_z2_value.values()[0] * obj_index_2_weight[0] + \
                point_with_higher_z2_value.values()[1] * obj_index_2_weight[1]
            point_obj_value = point.values()[0] * obj_index_2_weight[0] + point.values()[1] * obj_index_2_weight[1]
            # do not include the new extreme supported point if it close to the edge between the adjacent points, that 
            # is, the obj function value does not improve much.
            if math.isclose(point_obj_value, left_extreme_point_obj_value, rel_tol=self._obj_rel_tol):
                continue
            self._extreme_supported_nondominated_points.append(point)
            point_pairs_two_check.append(
                PointPair(point_with_higher_z1_value=point_with_higher_z1_value, point_with_higher_z2_value=point))
            point_pairs_two_check.append(
                PointPair(point_with_higher_z1_value=point, point_with_higher_z2_value=point_with_higher_z2_value))
        # sort the nondominated points in non-decreasing order of z1 values
        self._extreme_supported_nondominated_points = BolpDichotomicSearchWithGurobiSolver._sort_points(
            self._extreme_supported_nondominated_points, lambda x: x.z1())