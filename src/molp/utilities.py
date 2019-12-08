"""Implements utilities for solving a Multi-Objective Linear Program"""

from gurobipy import GRB


class ModelQueryUtilities:

    """Implements model query utilities"""

    @staticmethod
    def query_optimal_objective_values(model, solver_stage=None):
        """Queries the model for a feasible solution, and returns the objective values of the best solution found"""
        status = model.getAttr("Status")
        if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
            message = "the optimization call for the '%s' model ended with the '%s' status" % (
                    model.getAttr("ModelName"), status.value)
            if solver_stage:
                message = " ".join([message, "in the '%s' stage" % solver_stage])
            raise RuntimeError(message)
        values = []
        for i in range(model.getAttr("NumObj")):
            values.append(model.getObjective(index=i).getValue())
        return values