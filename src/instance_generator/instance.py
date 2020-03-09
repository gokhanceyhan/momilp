"""Implements the utilities to generate general multi-objective mixed-integer linear program instances

Referenced articles: 

@article{mavrotas2005multi,
  title={Multi-criteria branch and bound: A vector maximization algorithm for mixed 0-1 multiple objective linear programming},
  author={Mavrotas, George and Diakoulaki, Danae},
  journal={Applied mathematics and computation},
  volume={171},
  number={1},
  pages={53--71},
  year={2005},
  publisher={Elsevier}
}
@article{boland2015criterion,
  title={A criterion space search algorithm for biobjective mixed integer programming: The triangle splitting method},
  author={Boland, Natashia and Charkhgard, Hadi and Savelsbergh, Martin},
  journal={INFORMS Journal on Computing},
  volume={27},
  number={4},
  pages={597--618},
  year={2015},
  publisher={INFORMS}
}"""

from gurobipy import GRB, LinExpr, Model
import numpy as np
import os
import pandas as pd


class MomilpInstanceParameterSet:

    """Implements MOMILP instance parameter set"""

    def __init__(
            self,
            constraint_coeff_range=(-1, 20),
            continuous_var_obj_coeff_range=(-10, 10),
            integer_var_obj_coeff_range=(-200, 200),
            num_binary_vars=10,
            num_constraints=20,
            num_continuous_vars=10,
            num_discrete_objs=1,
            num_integer_vars=10,
            num_objs=3,
            rhs_range=(50, 100)):
        self.constraint_coeff_range = constraint_coeff_range
        self.continuous_var_obj_coeff_range = continuous_var_obj_coeff_range
        self.integer_var_obj_coeff_range = integer_var_obj_coeff_range
        self.num_binary_vars = num_binary_vars
        self.num_constraints = num_constraints
        self.num_continuous_vars = num_continuous_vars
        self.num_discrete_objs = num_discrete_objs
        self.num_integer_vars = num_integer_vars
        self.num_objs = num_objs
        self.rhs_range = rhs_range

    def to_dict(self):
        """Returns the dictionary representation of the parameter set"""
        return self.__dict__


class MomilpInstanceData:

    """Implements a MOMILP instance data
    
    NOTE: Based on the data generation schema defined in Mavrotas and Diakoulaki (2005) and Boland et al. (2015)"""

    def __init__(self, param_2_value, np_rand_num_generator_seed=0):
        np.random.seed(np_rand_num_generator_seed)
        self._constraint_coeff_df = None
        self._continuous_var_obj_coeff_df = None
        self._integer_var_obj_coeff_df = None
        self._param_2_value = param_2_value
        self._rhs = None
        self._create()

    def _create(self):
        """Creates the data"""
        self._create_constraint_coeff_df()
        self._create_continuous_var_obj_coeff_df()
        self._create_integer_var_obj_coeff_df()
        self._create_rhs()

    def _create_constraint_coeff_df(self):
        """Create the data frame of constraint coefficients"""
        num_constraints = self._param_2_value["num_constraints"]
        num_continuous_vars = self._param_2_value["num_continuous_vars"]
        num_integer_vars = self._param_2_value["num_integer_vars"]
        (low, high) = self._param_2_value["constraint_coeff_range"]
        continuous_var_columns = [i for i in range(num_continuous_vars)]
        integer_var_columns = [len(continuous_var_columns) + i for i in range(num_integer_vars)]
        continuous_var_constraint_df = pd.DataFrame(
            np.random.random_integers(low, high, size=(num_constraints - 1, num_continuous_vars))).append(
            pd.DataFrame(np.zeros(shape=(1, num_continuous_vars)))).reset_index(drop=True)
        continuous_var_constraint_df.columns = continuous_var_columns
        integer_var_constraint_df = pd.DataFrame(
            np.diag(np.random.random_integers(low, high=high, size=num_integer_vars))).append(
            pd.DataFrame(np.zeros(shape=(num_constraints - num_integer_vars -1, num_integer_vars)))).append(
            pd.DataFrame(np.ones(shape=(1, num_integer_vars)))).reset_index(drop=True)
        integer_var_constraint_df.columns = integer_var_columns
        self._constraint_coeff_df = pd.concat([continuous_var_constraint_df, integer_var_constraint_df], axis=1)

    def _create_continuous_var_obj_coeff_df(self):
        """Create the data frame of the objective function coefficients of the continuous variables"""
        num_objs = self._param_2_value["num_objs"]
        num_discrete_objs = self._param_2_value["num_discrete_objs"]
        discrete_obj_indices = [i for i in range(num_discrete_objs)]
        num_vars = self._param_2_value["num_continuous_vars"]
        (low, high) = self._param_2_value["continuous_var_obj_coeff_range"]
        df = pd.DataFrame(np.random.random_integers(low, high, size=(num_vars, num_objs)))
        df.iloc[:, discrete_obj_indices] = 0
        self._continuous_var_obj_coeff_df = df

    def _create_integer_var_obj_coeff_df(self):
        """Create the data frame of the objective function coefficients of the integer variables"""
        num_objs = self._param_2_value["num_objs"]
        num_vars = self._param_2_value["num_integer_vars"]
        (low, high) = self._param_2_value["integer_var_obj_coeff_range"]
        self._integer_var_obj_coeff_df = pd.DataFrame(
            np.random.random_integers(low, high, size=(num_vars, num_objs)))

    def _create_rhs(self):
        """Creates the series of the right-hand-side values of the constraints"""
        num_constraints = self._param_2_value["num_constraints"]
        num_integer_vars = self._param_2_value["num_integer_vars"]
        (low, high) = self._param_2_value["rhs_range"]
        self._rhs = pd.Series(np.random.random_integers(low, high=high, size=num_constraints - 1)).append(
            pd.Series(num_integer_vars / 3)).reset_index(drop=True)

    def constraint_coeff_df(self):
        """Returns the constraint coefficient data frame
        
        NOTE: An (m by n) matrix where rows are constraints and columns are variables"""
        return self._constraint_coeff_df

    def continuous_var_obj_coeff_df(self):
        """Returns the objective functions coefficients data frame for the continuous variables
        
        NOTE: An (m by n) matrix where rows are variables and columns are objective functions"""
        return self._continuous_var_obj_coeff_df

    def integer_var_obj_coeff_df(self):
        """Returns the objective functions coefficients data frame for the integer variables
        
        NOTE: An (m by n) matrix where rows are variables and columns are objective functions"""
        return self._integer_var_obj_coeff_df

    def rhs(self):
        """Returns the right-hand-side values of the constraints
        
        NOTE: A series of length m"""
        return self._rhs


class GurobiMomilpInstance:

    """Implements a MOMIILP instance
    
    NOTE: Based on the model defined in Mavrotas and Diakoulaki (2005) and Boland et al. (2015)"""

    _CONSTRAINT_NAME_FORMAT = "con_{index}"
    _CONTINUOUS_VARIABLE_NAME_FORMAT = "x_[{index}]"
    _INSTANCE_NAME_FORMAT = \
        "momilp_{num_objs}obj_{num_constraints}con_{num_integer_vars}int_{num_binary_vars}bin_{instance_number}.lp"
    _INTEGER_VARIABLE_NAME_FORMAT = "y_[{index}]"
    _OBJECTIVE_FUNCTION_NAME_FORMAT = "z_{index}"

    def __init__(self, param_2_value, instance_number=1, np_rand_num_generator_seed=0):
        self._data = MomilpInstanceData(param_2_value, np_rand_num_generator_seed=np_rand_num_generator_seed)
        self._instance_number = instance_number
        self._model = Model()
        self._param_2_value = param_2_value
        self._create()

    def _create(self):
        """Creates the model"""
        self._create_variables()
        self._create_objective_functions()
        self._create_constraints()

    def _create_constraints(self):
        """Creates the constraints of the model"""
        model = self._model
        num_constraints = self._param_2_value["num_constraints"]
        constraint_df = self._data.constraint_coeff_df()
        vars = model.getVars()
        rhs = self._data.rhs()
        for constraint_index in range(num_constraints):
            lhs = LinExpr()
            coeffs = constraint_df.iloc[constraint_index, :]
            lhs.addTerms(coeffs, vars)
            name = GurobiMomilpInstance._CONSTRAINT_NAME_FORMAT.format(index=constraint_index)
            model.addConstr(lhs, sense=GRB.LESS_EQUAL, rhs=rhs[constraint_index], name=name)
        model.update()
    
    def _create_objective_functions(self):
        """Creates the objective functions of the model"""
        model = self._model
        num_objs=self._param_2_value["num_objs"]
        vars = model.getVars()
        obj_coeff_df = pd.concat([self._data.continuous_var_obj_coeff_df(), self._data.integer_var_obj_coeff_df()])
        for obj_index in range(num_objs):
            expr = LinExpr()
            coeffs = list(obj_coeff_df.iloc[:, obj_index])
            expr.addTerms(coeffs, vars)
            name = GurobiMomilpInstance._OBJECTIVE_FUNCTION_NAME_FORMAT.format(index=obj_index)
            model.setObjectiveN(expr, obj_index, priority=num_objs-obj_index-1, name=name)
        model.update()

    def _create_variables(self):
        """Creates the variables of the model"""
        model = self._model
        num_binary_vars = self._param_2_value["num_binary_vars"]
        num_continuous_vars = self._param_2_value["num_continuous_vars"]
        num_integer_vars = self._param_2_value["num_integer_vars"]
        model.addVars(range(num_continuous_vars), lb=0.0, name="x")
        integer_var_indices = [i for i in range(num_integer_vars)]
        model.addVars(integer_var_indices[:num_binary_vars], vtype=GRB.BINARY, name="y")
        model.addVars(integer_var_indices[num_binary_vars:], vtype=GRB.INTEGER, name="y")
        model.update()

    def write(self, target_dir):
        """Writes the model in the target directory"""
        instance_name = GurobiMomilpInstance._INSTANCE_NAME_FORMAT.format(
            num_objs=self._param_2_value["num_objs"], 
            num_constraints=self._param_2_value["num_constraints"], 
            num_integer_vars=self._param_2_value["num_integer_vars"], 
            num_binary_vars=self._param_2_value["num_binary_vars"], 
            instance_number=self._instance_number)
        path = os.path.join(target_dir, instance_name)
        self._model.write(path)
