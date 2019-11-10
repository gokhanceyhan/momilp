"""Implements the momilp model"""

import abc
from gurobipy import GRB, Model, QuadExpr, read
from operator import itemgetter

class AbstractModel(metaclass=abc.ABCMeta):

    """Implements the abstract class for the momilp model"""

    @abc.abstractmethod
    def add_constraint(self, name, lhs, rhs, sense):
        """Adds constraint to the problem and return the created constraint object"""

    @abc.abstractmethod
    def copy(self):
        """Creates and returns a deep copy of the model"""

    @abc.abstractmethod
    def fix_integer_vector(self, y_bar):
        """Creates the constraint of 'y = y_bar'"""

    @abc.abstractmethod
    def int_vars(self):
        """Returns the integer decision variables in the problem"""

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
    def solve(self):
        """Solves the model"""

    @abc.abstractmethod
    def update_constraint(self, name, constraint):
        """Updates the constraint with the specified name"""

    @abc.abstractmethod
    def X(self):
        """Returns the feasible set of the problem"""

    @abc.abstractmethod
    def Z(self):
        """Returns the image of the feasible set of the problem in the criterion space"""


class GurobiMomilpModel(AbstractModel):

    """Implements momilp model by using Gurobi solver"""

    _MODEL_NAME = "P"

    def __init__(
            self, file_name, discrete_objective_indices=None, log_to_console=False, log_to_file=True, num_obj=None):
        self._model = read(file_name)
        self._set_params(
            log_to_console=log_to_console, log_to_file=log_to_file, model_name=GurobiMomilpModel._MODEL_NAME)
        self._validate(discrete_objective_indices=discrete_objective_indices, num_obj=num_obj)
        self._constraint_name_2_constraint = {}
        self._objective_name_2_objective = {}
        self._initialize()

    def _initialize(self):
        """Creates the constraints in the problem"""
        model = self._model
        for constraint in model.getConstrs():
            self._constraint_name_2_constraint[constraint.getAttr("ConstrName")] = constraint
        for obj_index in range(model.getAttr("NumObj")):
            model.setParam("ObjNumber", obj_index)
            obj = model.getObjective()
            name = model.getAttr("ObjNName")
            self._objective_name_2_objective[name] = obj
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

    def _validate(self, discrete_objective_indices=None, num_obj=None):
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
        if num_obj and num_obj != model_num_obj:
            message = "The model '%s' has '%s' objectives whereas the specified value is '%s'" % (
                model_name, model_num_obj, num_obj)
            raise ValueError(message)
        objective_index_and_priorities = []
        for obj_index in range(model.getAttr("NumObj")):
            model.setParam("ObjNumber", obj_index)
            obj = model.getObjective()
            priority = model.getAttr("ObjNPriority")
            objective_index_and_priorities.append((obj_index, priority))
        objective_index_and_priorities = sorted(objective_index_and_priorities, key=itemgetter(1), reverse=True)
        top_objective_index = objective_index_and_priorities[0][0]
        if discrete_objective_indices and top_objective_index not in discrete_objective_indices:
            message = "The highest priority objective '%s' is not in the set of specified discrete objectives '%s'" % (
                top_objective_index, discrete_objective_indices)
            raise ValueError(message)
    
    def add_constraint(self, lhs, name, rhs, sense):
        try:
            assert not isinstance(lhs, QuadExpr)
            assert not isinstance(rhs, QuadExpr)
        except AssertionError as error:
            message = "The constraint '%s' cannot be added to the linear model '%s'" % (
                name, self._model.getAttr("ModelName"))
            raise RuntimeError(message) from error
        return self._model.addLConstr(lhs, sense=sense, rhs=rhs, name=name)

    def copy(self):
        return self._model.copy()

    def fix_integer_vector(self, y_bar):
        int_vars = self.int_vars()
        for int_var, y_bar_ in zip(int_vars, y_bar):
            int_var.setAttr("LB", y_bar_)
            int_var.setAttr("UB", y_bar_)

    def int_vars(self):
        vars_ = self._model.getVars()
        return [var for var in vars_ if var.getAttr("VType") == "B" or var.getAttr("VType") == "I"]

    def num_int_vars(self):
        return self._model.getAttr("NumIntVars")

    def num_obj(self):
        return self._model.getAttr("NumObj")

    def objective(self, index):
        return self._model.getObjective(index=index)

    def problem(self):
        return self._model

    def solve(self):
        self._model.optimize()

    def update_constraint(self, name):
        pass

    def X(self):
        return list(self._constraint_name_2_constraint.values())

    def Z(self):
        return list(self._objective_name_2_objective.values())
