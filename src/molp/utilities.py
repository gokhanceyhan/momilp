"""Implements utilities for solving a Multi-Objective Linear Program"""

from gurobipy import GRB


class ModelQueryUtilities:

    """Implements model query utilities"""

    @staticmethod
    def query_optimal_objective_values(model, raise_error_if_infeasible=True, solver_stage=None):
        """Queries the model for a feasible solution, and returns the objective values of the best solution found"""
        status = model.getAttr("Status")
        values = []
        if status in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED]:
            message = "the optimization call for the '%s' model ended with the '%s' status" % (
                    model.getAttr("ModelName"), status)
            if solver_stage:
                message = " ".join([message, "in the '%s' stage" % solver_stage])
            if status == GRB.INFEASIBLE and not raise_error_if_infeasible:
                return values, status
            raise RuntimeError(message)
        for i in range(model.getAttr("NumObj")):
            values.append(model.getObjective(index=i).getValue())
        return values, status