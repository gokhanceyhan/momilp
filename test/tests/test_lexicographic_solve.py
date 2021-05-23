"""Implements tests for the lexicographic solution of a multi-objective model with Gurobi"""

from hamcrest import assert_that, is_
from gurobipy import read
import os
from unittest import main, TestCase

from src.common.elements import OptimizationStatus
from src.momilp.utilities import ModelQueryUtilities

class GurobiLexicographicSolveTest(TestCase):

    """Implements tests for the Gurobi lexicographic solve"""

    def setUp(self):
        self._test_data_dir = os.path.join(os.environ["PYTHON_TEST_PATH"], "data") 
        self.assert_that = assert_that

    def test_unique_objective_priorities(self):
        """Tests the lexicographic solution of a 3-obj problem with unique objective priorities"""
        file_name = os.path.join(self._test_data_dir, "three_obj_blp.lp")
        model = read(file_name)
        model.optimize()
        gurobi_status = model.getAttr("Status")
        optimization_status = ModelQueryUtilities.GUROBI_STATUS_2_OPTIMIZATION_STATUS[gurobi_status]
        self.assert_that(optimization_status, is_(OptimizationStatus.OPTIMAL))


if __name__ == '__main__':
    main()
