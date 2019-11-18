"""Implements algorithm to solve the momilp"""

import abc
from enum import Enum


class AbstractAlgorithm(metaclass=abc.ABCMeta):

    """Implements abstract class for the algorithm"""
    
    @abc.abstractmethod
    def run(self):
        """Runs the algorithm"""


class AlgorithmType(Enum):

    """Implements algorithm type"""

    CONE_BASED_SEARCH = "cone-based search"


class ConeBasedSearchAlgorithm(AbstractAlgorithm):

    """Implements the cone-based search algorithm"""

    def __init__(self, model):
        self._model = model
        self._initialize()

    def _initialize(self):
        """Initializes the algorithm"""

    def run(self):
        pass


class Factory:

    """Implements algorithm factory to solve momilp"""

    _SUPPORTED_ALGORITHM_TYPES = [AlgorithmType.CONE_BASED_SEARCH]
    _SUPPORTED_NUM_OBJECTIVES = [3]
    _UNSUPPORTED_ALGORITHM_TYPE_ERROR_MESSAGE = \
        "the '{type!s}' algorithm type is not supported, select one of the '{supported_types!s}' types"
    _UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE = \
        "the '{num_obj!s}'-obj problem is not supported, select one of the '{supported_num_obj!s}' values"

    @staticmethod
    def _create_cone_based_search_algorithm(model):
        """Creates and returns the cone-based search algorithm"""
        return ConeBasedSearchAlgorithm(model)

    @staticmethod
    def create(model, algorithm_type=AlgorithmType.CONE_BASED_SEARCH):
        """Creates algorithm"""
        num_obj = model.NumObj()
        if num_obj not in Factory._SUPPORTED_NUM_OBJECTIVES:
            error_message = Factory._UNSUPPORTED_NUM_OBJECTIVES_ERROR_MESSAGE.format(
                num_obj=num_obj, supported_num_obj=Factory._SUPPORTED_NUM_OBJECTIVES)
            raise ValueError(error_message)
        if algorithm_type not in Factory._SUPPORTED_ALGORITHM_TYPES:
            error_message = Factory._UNSUPPORTED_ALGORITHM_TYPE_ERROR_MESSAGE.format(
                type=algorithm_type, supported_types=Factory._SUPPORTED_ALGORITHM_TYPES)
            raise ValueError(error_message)
        if algorithm_type == AlgorithmType.CONE_BASED_SEARCH:
            return Factory._create_cone_based_search_algorithm(model)
