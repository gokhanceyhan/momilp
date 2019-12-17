"""Implements the momilp model"""

import abc
import copy
from gurobipy import GRB, LinExpr, Model, QuadExpr, read
from operator import itemgetter
from src.common.elements import SolverStage
from src.momilp.utilities import ModelQueryUtilities


class AbstractModel(metaclass=abc.ABCMeta):

    """Implements the abstract class for the momilp model"""

    @abc.abstractmethod
    def add_constraint(self, name, lhs, rhs, sense):
        """Adds constraint to the problem and return the created constraint object"""

    @abc.abstractmethod
    def copy(self):
        """Creates and returns a deep copy of the object"""

    @abc.abstractmethod
    def fix_integer_vector(self, y_bar):
        """Creates the constraint of 'y = y_bar'"""

    @abc.abstractmethod
    def num_int_vars(self):
        """Returns the number of integer decision variables in the problem"""

    @abc.abstractmethod
    def num_obj(self):
        """Returns the number of objectives"""

    @abc.abstractmethod
    def objective(self, index):
        """Returns the objective having the specified index
        
        Note: Objective indices start from zero."""

    @abc.abstractmethod
    def problem(self):
        """Returns the problem as a math model"""

    @abc.abstractmethod
    def relax(self):
        """Removes the integrality constraints from the model"""

    @abc.abstractmethod
    def remove_constraint(self, constraint_names):
        """Removes the specified constraints from the model"""

    @abc.abstractmethod
    def solve(self):
        """Solves the model"""

    @abc.abstractmethod
    def update_constraint(self, name, constraint):
        """Updates the constraint with the specified name"""

    @abc.abstractmethod
    def update_model(self):
        """Updates the model"""

    @abc.abstractmethod
    def X(self):
        """Returns the feasible set of the problem as a dict of constraint name to constraint"""

    @abc.abstractmethod
    def y(self):
        """Returns the integer decision variables in the problem"""

    @abc.abstractmethod
    def Z(self):
        """Returns the image of the feasible set of the problem in the criterion space as a dict of objective name to 
        objective var"""

    @abc.abstractmethod
    def write(self, file_name):
        """Writes the model"""


class GurobiMomilpModel(AbstractModel):

    """Implements momilp model by using Gurobi solver"""

    _MODEL_NAME = "P"

    def __init__(
            self, discrete_objective_indices=None, file_name=None, gurobi_model=None, log_to_console=False, 
            log_to_file=True, num_obj=None, scale=True):
        self._constraint_name_2_constraint = {}
        self._discrete_objective_indices = discrete_objective_indices or []
        self._int_var_2_original_lb_and_ub = {}
        assert file_name or gurobi_model, "either file name or gurobi model must be given"
        self._model = gurobi_model or read(file_name)
        self._num_obj = num_obj
        self._objective_index_2_priority = {}
        self._objective_name_2_range = {}
        self._objective_name_2_scaler = {}
        self._objective_name_2_variable = {}
        self._primary_objective_index = None
        self._region_defining_constraint_names = []
        self._tabu_constraint_names = []
        self._y = []
        self._set_params(
            log_to_console=log_to_console, log_to_file=log_to_file, model_name=GurobiMomilpModel._MODEL_NAME)
        self._validate()
        self._initialize()
        if scale:
            self._scale_model()

    def _initialize(self):
        """Creates the constraints in the problem"""
        model = self._model
        for constraint in model.getConstrs():
            self._constraint_name_2_constraint[constraint.getAttr("ConstrName")] = constraint
        for obj_index in range(model.getAttr("NumObj")):
            model.setParam("ObjNumber", obj_index)
            name = model.getAttr("ObjNName")
            obj = model.getObjective(index=obj_index)
            # define continuous variables for the objective functions
            obj_var = model.addVar(name=name)
            self.add_constraint(
                obj - obj_var if model.getAttr("ModelSense") == GRB.MAXIMIZE else obj + obj_var, name, 0.0, GRB.EQUAL)
            self._objective_name_2_variable[name] = obj_var
        # store the integer variable vector
        vars_ = self._model.getVars()
        self._y = [var for var in vars_ if var.getAttr("VType") == "B" or var.getAttr("VType") == "I"]
        # save the original bounds of the integer variables
        self._int_var_2_original_lb_and_ub = {var: (var.LB, var.UB) for var in self._y}
        model.update()

    def _scale_model(self):
        """Scales the model"""
        model = self._model
        sense = model.getAttr("ModelSense")
        # Implement the procedure to transform the feasible objective space into R_{>=0}
        max_priority = 0
        for index in range(self._num_obj):
            model.setParam("ObjNumber", index)
            priority = model.getAttr("ObjNPriority")
            max_priority = max(max_priority, priority)
        for index in range(self._num_obj):
            model.setParam("ObjNumber", index)
            priority = model.getAttr("ObjNPriority")
            # make the objective as the highest priority objective
            model.setAttr("ObjNPriority", max_priority + 1)
            # maximize
            model.setAttr("ModelSense", -1)
            model.optimize()
            max_point_sol = ModelQueryUtilities.query_optimal_solution(
                model, raise_error_if_infeasible=True, 
                solver_stage=SolverStage.MODEL_SCALING).point_solution()
            # minimize
            model.setAttr("ModelSense", 1)
            model.optimize()
            min_point_sol = ModelQueryUtilities.query_optimal_solution(
                model, raise_error_if_infeasible=True, 
                solver_stage=SolverStage.MODEL_SCALING).point_solution()
            obj_name = model.getAttr("ObjNName")
            self._objective_name_2_range[obj_name] = ObjectiveRange(max_point_sol, min_point_sol)
            
            def obj_scaler(value, obj_min=min_point_sol.point().values()[index]):
                return value + max(-1 * obj_min, 0)

            self._objective_name_2_scaler[obj_name] = obj_scaler
            # restore the original priority of the objective
            model.setAttr("ObjNPriority", priority)
        # restore the original objective sense
        model.setAttr("ModelSense", sense)
        model.update()

    def _set_params(self, log_to_console=False, log_to_file=True, model_name=None):
        """Sets the model parameters"""
        model = self._model
        if not log_to_console:
            model.setParam("LogToConsole", 0)
        if not log_to_file:
            model.setParam("LogFile", "")
        model.setAttr("ModelName", GurobiMomilpModel._MODEL_NAME)
        model.update()

    def _validate(self):
        """Validates the model"""
        model = self._model
        model_name = self._model.getAttr("ModelName")
        try:
            assert model.isMIP
            assert model.isMultiObj
            assert not model.isQP
            assert not model.isQCP
        except AssertionError as error:
            message = "The model '%s' is not a multi-objective mixed-integer linear program" % model_name
            raise RuntimeError(message) from error
        model_num_obj = model.getAttr("NumObj")
        num_obj = self._num_obj
        if num_obj and num_obj != model_num_obj:
            message = "The model '%s' has '%s' objectives whereas the specified value is '%s'" % (
                model_name, model_num_obj, num_obj)
            raise ValueError(message)
        self._num_obj = num_obj or model_num_obj
        objective_index_and_priorities = []
        for obj_index in range(self._num_obj):
            model.setParam("ObjNumber", obj_index)
            priority = model.getAttr("ObjNPriority")
            objective_index_and_priorities.append((obj_index, priority))
            self._objective_index_2_priority[obj_index] = priority
        unique_priority_values = set(priority for (_, priority) in objective_index_and_priorities)
        if len(unique_priority_values) < model_num_obj:
            message = "The model objective functions must have different priorities, '%s" % unique_priority_values
            raise ValueError(message)
        objective_index_and_priorities = sorted(objective_index_and_priorities, key=itemgetter(1), reverse=True)
        top_objective_index = objective_index_and_priorities[0][0]
        discrete_objective_indices = self._discrete_objective_indices
        if not discrete_objective_indices:
            # identify the list of objectives with integer variables only
            for obj_index in range(self._num_obj):
                obj = model.getObjective(index=obj_index)
                obj_vars = [obj.getVar(term_idx) for term_idx in range(obj.size())]
                all_integer = all([var.getAttr("VType") == "B" or var.getAttr("VType") == "I" for var in obj_vars])
                if not all_integer:
                    continue
                discrete_objective_indices.append(obj_index)
        if not discrete_objective_indices:
            message = "At least one objective function must have discrete feasible set"
            raise ValueError(message)
        if discrete_objective_indices and top_objective_index not in discrete_objective_indices:
            message = "The highest priority objective '%s' is not in the set of discrete objectives '%s'" % (
                top_objective_index, discrete_objective_indices)
            raise ValueError(message)
        self._primary_objective_index = discrete_objective_indices[0]
    
    def add_constraint(self, lhs, name, rhs, sense, region_constraint=False, tabu_constraint=False):
        try:
            assert not isinstance(lhs, QuadExpr)
            assert not isinstance(rhs, QuadExpr)
        except AssertionError as error:
            message = "The constraint '%s' cannot be added to the linear model '%s'" % (
                name, self._model.getAttr("ModelName"))
            raise RuntimeError(message) from error
        constraint = self._model.addLConstr(lhs, sense=sense, rhs=rhs, name=name)
        self._constraint_name_2_constraint[name] = constraint
        if region_constraint:
            self._region_defining_constraint_names.append(name)
        if tabu_constraint:
            self._tabu_constraint_names.append(name)
        return constraint

    def binary(self):
        """Returns True if this is a binary-integer model, False otherwise."""
        y = self._y
        for y_ in y:
            if y_.LB < 0 or y_.UB > 1:
                return False
        return True

    def change_objective_priorities(self, obj_num_2_priority):
        """Changes the priorities of the objectives based on the given objective number to priority dictionary"""
        model = self._model
        for obj_num, priority in obj_num_2_priority:
            model.setParam("ObjNumber", obj_num)
            priority = model.setAttr("ObjNPriority", priority)

    def constraint_name_2_constraint(self):
        """Returns the constraint name to constraint"""
        return self._constraint_name_2_constraint
    
    def copy(self):
        return copy.deepcopy(self)

    def fix_integer_vector(self, y_bar):
        y = self._y
        for y_, y_bar_ in zip(y, y_bar):
            y_.setAttr("LB", y_bar_)
            y_.setAttr("UB", y_bar_)
    
    def num_int_vars(self):
        return self._model.getAttr("NumIntVars")

    def num_obj(self):
        return self._num_obj

    def objective(self, index):
        return self._model.getObjective(index=index)

    def objective_index_2_priority(self):
        """Returns the dictionary of objective index to priority"""
        return self._objective_index_2_priority

    def objective_name_2_range(self):
        """Returns the objective name to objective range"""
        return self._objective_name_2_range

    def objective_name_2_scaler(self):
        """Returns the objective name to scaler"""
        return self._objective_name_2_scaler

    def primary_objective_index(self):
        """Returns the primary objective index"""
        return self._primary_objective_index

    def problem(self):
        return self._model

    def region_defining_constraint_names(self):
        """Returns the region defining constraint names"""
        return self._region_defining_constraint_names

    def relax(self):
        # NOTE: I do not use 'model.relax()' here as it removes the references to the y-vector
        for y_ in self._y:
            y_.setAttr("VType", "C")

    def remove_constraint(self, constraint_names):
        constraint_names = constraint_names if isinstance(constraint_names, list) else [constraint_names]
        model = self._model
        constraints_to_remove = [
            self._constraint_name_2_constraint[constraint_name] for constraint_name in constraint_names]
        model.remove(constraints_to_remove)
        for constraint_name in constraint_names:
            del self._constraint_name_2_constraint[constraint_name]
            if constraint_name in self._region_defining_constraint_names:
                self._region_defining_constraint_names.remove(constraint_name)
            if constraint_name in self._tabu_constraint_names:
                self._tabu_constraint_names.remove(constraint_name)

    def restore_original_bounds_of_integer_variables(self):
        """Removes the posteriori-added bounds on the y-vector if there exist any"""
        y = self._y
        for y_ in y:
            y_.setAttr("LB", self._int_var_2_original_lb_and_ub[y_][0])
            y_.setAttr("UB", self._int_var_2_original_lb_and_ub[y_][1])

    def solve(self):
        self._model.optimize()

    def tabu_constraint_names(self):
        """Returns the tabu constraint names"""
        return self._tabu_constraint_names

    def update_constraint(self, name):
        pass

    def update_model(self):
        self._model.update()

    def X(self):
        return self._constraint_name_2_constraint

    def y(self):
        return self._y

    def Z(self):
        return self._objective_name_2_variable

    def write(self, file_name):
        self._model.write(file_name)


class ObjectiveRange:

    """Implements objective range"""

    def __init__(self, max_point_solution, min_point_solution):
        self._max_point_solution = max_point_solution
        self._min_point_solution = min_point_solution

    def max_point_solution(self):
        """Returns the max point solution"""
        return self._max_point_solution

    def min_point_solution(self):
        """Returns the min point solution"""
        return self._min_point_solution