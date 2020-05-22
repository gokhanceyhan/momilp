"""Implements the executor class to run the momilp solver"""

import argparse
from enum import Enum
from gurobipy import Model, read
import logging
import os
import pandas as pd
import time
from src.common.elements import SolverPackage
from src.momilp.algorithm import AlgorithmFactory
from src.momilp.model import GurobiMomilpModel
from src.momilp.utilities import ReportCreator

logging.getLogger().setLevel(logging.INFO)


class ExecutionStatistics:

    """Implements execution statistics of an algorithm on an instance"""

    def __init__(
            self, algorithm_name, instance_name, elapsed_time_in_seconds=0, 
            elapsed_time_in_seconds_for_dominance_test_problem=0, 
            elapsed_time_in_seconds_for_search_problem=0, 
            elapsed_time_in_seconds_for_slice_problem=0, num_dominance_test_model_solved=0, 
            num_efficient_integer_vectors=0, 
            num_iterations=0, num_milp_solved=0, num_nd_edges=0, num_nd_points=0):
        self._algorithm_name = algorithm_name
        self._elapsed_time_in_seconds = elapsed_time_in_seconds
        self._elapsed_time_in_seconds_for_dominance_test_problem = elapsed_time_in_seconds_for_dominance_test_problem
        self._elapsed_time_in_seconds_for_search_problem = elapsed_time_in_seconds_for_search_problem
        self._elapsed_time_in_seconds_for_slice_problem = elapsed_time_in_seconds_for_slice_problem
        self._instance_name = instance_name
        self._num_dominance_test_model_solved = num_dominance_test_model_solved
        self._num_efficient_integer_vectors = num_efficient_integer_vectors
        self._num_iterations = num_iterations
        self._num_milp_solved = num_milp_solved
        self._num_nd_edges = num_nd_edges
        self._num_nd_points = num_nd_points

    def algorithm_name(self):
        """Returns the name of the algorithm"""
        return self._algorithm_name
    
    def elapsed_time_in_seconds(self):
        """Returns the elapsed time in seconds"""
        return self._elapsed_time_in_seconds

    def elapsed_time_in_seconds_for_dominance_test_problem(self):
        """Returns the elapsed time in seconds in dominance test problem solving"""
        return self._elapsed_time_in_seconds_for_dominance_test_problem

    def elapsed_time_in_seconds_for_search_problem(self):
        """Returns the elapsed time in seconds in search problem solving"""
        return self._elapsed_time_in_seconds_for_search_problem

    def elapsed_time_in_seconds_for_slice_problem(self):
        """Returns the elapsed time in seconds in slice problem solving"""
        return self._elapsed_time_in_seconds_for_slice_problem

    def instance_name(self):
        """Returns the name of the instance"""
        return self._instance_name

    def num_dominance_test_model_solved(self):
        """Returns the number of dominance test models solved"""
        return self._num_dominance_test_model_solved

    def num_efficient_integer_vectors(self):
        """Returns the number of efficient integer vectors generated"""
        return self._num_efficient_integer_vectors

    def num_iterations(self):
        """Returns the number of iterations"""
        return self._num_iterations

    def num_milp_solved(self):
        """Returns the number of milp models solved"""
        return self._num_milp_solved

    def num_nd_edges(self):
        """Returns the number of nondominated edges"""
        return self._num_nd_edges

    def num_nd_points(self):
        """Returns the number of nondominated points"""
        return self._num_nd_points

    def to_dict(self):
        return {
            "algorithm": self._algorithm_name,
            "instance": self._instance_name,
            "y_eff": self._num_efficient_integer_vectors,
            "iter": self._num_iterations,
            "num_milp": self._num_milp_solved,
            "nd_edges": self._num_nd_edges,
            "nd_points": self._num_nd_points,
            "time (sec)": self._elapsed_time_in_seconds,
            "time_search (sec)": self._elapsed_time_in_seconds_for_search_problem,
            "time_slice (sec)": self._elapsed_time_in_seconds_for_slice_problem,
            "time_dom_test (sec)": self._elapsed_time_in_seconds_for_dominance_test_problem,
            "num_dom_lp": self._num_dominance_test_model_solved
        }


class Executor:

    """Implements the momilp solver executor"""

    _EXECUTION_STATISTICS_REPORT_FILE_NAME = "summary.csv"
    _MODEL_CLASS_TO_SOLVER_PACKAGE = {Model: SolverPackage.GUROBI}
    _NUM_DECIMALS_FOR_TIME_IN_SECONDS = 2
    _SUPPORTED_SOLVER_PACKAGES = [SolverPackage.GUROBI]
    _UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE = \
        "the solver package is not supported, define the model in one of the '{supported_solvers!s}' solver packages"

    def __init__(
            self, model_files, dichotomic_search_rel_tol=1e-6, discrete_objective_indices=None, 
            explore_decision_space=True, max_num_iterations=None, obj_index_2_range=None, rel_coverage_gap=0.0, 
            search_num_threads=None, search_time_limit_in_seconds=None, solver_package=SolverPackage.GUROBI):
        self._dichotomic_search_rel_tol = dichotomic_search_rel_tol
        self._discrete_objective_indices = discrete_objective_indices
        self._explore_decision_space=explore_decision_space
        self._max_num_iterations = max_num_iterations
        self._model_files = model_files
        self._obj_index_2_range = obj_index_2_range or {}
        self._rel_coverage_gap = rel_coverage_gap
        self._search_num_threads = search_num_threads
        self._search_time_limit_in_seconds = search_time_limit_in_seconds
        self._statistics = []
        if solver_package not in Executor._SUPPORTED_SOLVER_PACKAGES:
            error_message = Executor._UNSUPPORTED_SOLVER_PACKAGE_ERROR_MESSAGE.format(
                solver=solver_package, supported_solvers=Executor._SUPPORTED_SOLVER_PACKAGES)
            raise ValueError(error_message)
        self._solver_package = solver_package

    def _collect_statistics(self, algorithm, elapsed_time_in_seconds, instance_name, state):
        """Collects the statistics for the algorithm and the instance"""
        algorithm_name = algorithm.__class__.__name__
        iterations = state.iterations()
        num_iterations = len(iterations)
        iteration_statistics = [iteration.statistics() for iteration in iterations]
        # model count statistics
        num_milp_solved = sum([s.num_milp_solved() for s in iteration_statistics])
        num_dominance_test_model_solved = algorithm.dominance_filter().num_models_solved()
        # elapsed time statistics
        time_precision = Executor._NUM_DECIMALS_FOR_TIME_IN_SECONDS
        elapsed_time_in_seconds_for_search_problem = round(
            sum([s.elapsed_time_in_seconds_for_search_problem() for s in iteration_statistics]), time_precision)
        elapsed_time_in_seconds_for_slice_problem = round(
            sum([s.elapsed_time_in_seconds_for_slice_problem() for s in iteration_statistics]), time_precision)
        elapsed_time_in_seconds_for_dominance_test_problem = round(
            algorithm.dominance_filter().elapsed_time_in_seconds(), time_precision)
        # solution state statistics
        solution_state = state.solution_state()
        num_nd_edges = Executor._num_nondominated_edges(solution_state)
        num_nd_points = Executor._num_nondominated_points(solution_state)
        num_efficient_integer_vectors = len(solution_state.efficient_integer_vectors())
        statistics = ExecutionStatistics(
            algorithm_name, instance_name, elapsed_time_in_seconds=elapsed_time_in_seconds, 
            elapsed_time_in_seconds_for_dominance_test_problem=elapsed_time_in_seconds_for_dominance_test_problem,
            elapsed_time_in_seconds_for_search_problem=elapsed_time_in_seconds_for_search_problem, 
            elapsed_time_in_seconds_for_slice_problem=elapsed_time_in_seconds_for_slice_problem, 
            num_dominance_test_model_solved=num_dominance_test_model_solved,
            num_efficient_integer_vectors=num_efficient_integer_vectors, num_iterations=num_iterations, 
            num_milp_solved=num_milp_solved, num_nd_edges=num_nd_edges, num_nd_points=num_nd_points)
        self._statistics.append(statistics)

    def _export_statistics(self, working_dir):
        """Exports the statistics to the working directory as a CSV file"""
        df = pd.DataFrame.from_records([statistics_.to_dict() for statistics_ in self._statistics])
        df.to_csv(os.path.join(working_dir, Executor._EXECUTION_STATISTICS_REPORT_FILE_NAME))

    @staticmethod
    def _num_nondominated_edges(solution_state):
        """Returns the number of nondominated edges"""
        edges = [nd_edge.edge() for nd_edge in solution_state.nondominated_edges()]
        return len(set(edges))

    @staticmethod
    def _num_nondominated_points(solution_state):
        """Returns the number of nondominated points"""
        points = [nd_point.point() for nd_point in solution_state.nondominated_points()]
        return len(set(points))

    def execute(self, working_dir):
        """Executes the momilp solver"""
        for model_file in self._model_files:
            start_time = time.time()
            algorithm = AlgorithmFactory.create(
                model_file, working_dir, dichotomic_search_rel_tol=self._dichotomic_search_rel_tol, 
                discrete_objective_indices=self._discrete_objective_indices, 
                explore_decision_space=self._explore_decision_space, max_num_iterations=self._max_num_iterations, 
                obj_index_2_range=self._obj_index_2_range, rel_coverage_gap=self._rel_coverage_gap, 
                search_num_threads=self._search_num_threads, 
                search_time_limit_in_seconds=self._search_time_limit_in_seconds)
            state = algorithm.run()
            elapsed_time_in_seconds = round(time.time() - start_time, Executor._NUM_DECIMALS_FOR_TIME_IN_SECONDS)
            instance_name = os.path.splitext(os.path.basename(model_file))[0]
            report_creator = ReportCreator(algorithm.momilp_model(), state, instance_name, working_dir)
            report_creator.create()
            self._collect_statistics(algorithm, elapsed_time_in_seconds, instance_name, state)
        self._export_statistics(working_dir)
        if algorithm.errors():
            logging.warning("\n".join(algorithm.errors()))


class MomilpSolverApp:

    """Implements the command line application for the momilp solver executor"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="momilp solver app")
        parser.add_argument(
            "-a", "--alpha", default=0.0, help="the shift coefficient used in separating the dominated space. This "
            "can be set to a positive value when an approximation of the nondominated frontier is sufficient for the "
            "use case instead of the exact nondominated set (the generated points are multiplied with (1 + alpha) "
            "while defining the dominated space boundary). The generated set is an approximation of the true "
            "nondominated set and may include dominated points. However, the generated set 'alpha-dominates' the "
            "nondominated set.")
        parser.add_argument(
            "-b", "--beta", default=1e-6, help="the stopping condition parameter for the dichotomic search while "
            "generating the frontier of a slice problem. An extreme supported ppoint will not be generated if its "
            "objective function value is not higher than (1 + beta) of the objective function value of the adjacent "
            "extreme supported points.")
        parser.add_argument(
            "-d", "--discrete-objective-indices", 
            help="the list of discrete objective indices in the model file, e.g. '0', or '0, 1, or '0, 1, 2'. This "
            "specifies the objectives that include continuous variables, but can only have finite feasible criterion "
            "set.")
        parser.add_argument(
            "-e", "--explore-decision-space", action='store_true', help="generate all efficient integer vectors")
        parser.add_argument("-i", "--iteration-limit", help="maximum nunmber of iterations to run")
        parser.add_argument(
            "-m", "--model-file-path", 
            help="sets the path to the directory where the model files (.lp format) are stored")
        parser.add_argument("-n", "--num-threads", help="sets the number of threads for the milp solver")
        parser.add_argument(
            "-s", "--solver-package", choices=[SolverPackage.GUROBI.value], help="sets the solver package to use")
        parser.add_argument("-t", "--time-limit", help="sets the time limit in seconds for the milp solver")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        model_file_path = args.model_file_path
        model_files = [os.path.join(model_file_path, f) for f in os.listdir(model_file_path) if f.endswith(".lp")]
        alpha = float(args.alpha)
        beta = float(args.beta)
        explore_decision_space = args.explore_decision_space
        if explore_decision_space and alpha > 0:
            raise ValueError("alpha value must be zero if the decision space is to be explored")
        discrete_objective_indices = [int(s) for s in args.discrete_objective_indices.split(",")] if \
            args.discrete_objective_indices else None
        max_num_iterations = int(args.iteration_limit) if args.iteration_limit else None
        num_threads = int(args.num_threads) if args.num_threads else None
        time_limit_in_seconds = int(args.time_limit) if args.time_limit else None
        executor = Executor(
            model_files, dichotomic_search_rel_tol=beta, discrete_objective_indices=discrete_objective_indices, 
            explore_decision_space=explore_decision_space, max_num_iterations=max_num_iterations, 
            rel_coverage_gap=alpha, solver_package=SolverPackage(args.solver_package), search_num_threads=num_threads, 
            search_time_limit_in_seconds=time_limit_in_seconds)
        executor.execute(args.working_dir)