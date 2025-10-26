# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 14 2025
@name:   Algorithm Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from abc import ABC, abstractmethod

from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Algorithm"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class AlgorithmType(Enum): Numeric, Array, Table = list(range(3))
class Algorithm(ABC, metaclass=RegistryMeta):
    Type = AlgorithmType

    @staticmethod
    @abstractmethod
    def algorithm(calculation, arguments, parameters, /, vartype): pass


class NumericAlgorithm(Algorithm, register=(AlgorithmType.Numeric, False)):
    @staticmethod
    def algorithm(calculation, arguments, parameters, /, vartype):
        assert all([isinstance(argument, np.number) for argument in arguments])
        return calculation(arguments, parameters)


class VectorizedArrayAlgorithm(Algorithm, register=(AlgorithmType.Array, True)):
    @staticmethod
    def algorithm(calculation, arguments, parameters, /, vartype):
        assert all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters.values()])
        function = lambda *variables, **constants: calculation(variables, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[vartype], vectorize=True)


class VectorizedTableAlgorithm(Algorithm, register=(AlgorithmType.Table, True)):
    @staticmethod
    def algorithm(calculation, arguments, parameters, /, vartype):
        assert all([isinstance(argument, (pd.Series, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters.values()])
        function = lambda variables, **constants: calculation(variables, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)


class UnVectorizedArrayAlgorithm(Algorithm, register=(AlgorithmType.Array, False)):
    @staticmethod
    def algorithm(calculation, arguments, parameters, /, vartype):
        assert all([isinstance(argument, (xr.DataArray, np.ndarray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters.values()])
        return calculation(arguments, parameters)


class UnVectorizedTableAlgorithm(Algorithm, register=(AlgorithmType.Table, False)):
    @staticmethod
    def algorithm(calculation, arguments, parameters, /, vartype):
        assert all([isinstance(argument, (pd.Series, np.ndarray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters.values()])
        return calculation(arguments, parameters)



