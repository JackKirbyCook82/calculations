# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 14 2025
@name:   Algorithm Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["NumericAlgorithm", "IntegralAlgorithm", "VectorizedArrayAlgorithm", "VectorizedTableAlgorithm", "UnVectorizedArrayAlgorithm", "UnVectorizedTableAlgorithm"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Algorithm(ABC):
    @staticmethod
    @abstractmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs): pass


class NumericAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, np.number) for argument in arguments])
        assert not any([isinstance(parameter, np.number) for parameter in parameters])
        return calculation(arguments, parameters)


class IntegralAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs):
        pass


class VectorizedArrayAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, vartype, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        function = lambda *variables, **constants: calculation(variables, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[vartype], vectorize=True)


class VectorizedTableAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, vartype, **kwargs):
        assert all([isinstance(argument, (pd.Series, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters])
        function = lambda variables, **constants: calculation(variables, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)


class UnVectorizedArrayAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, np.ndarray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        return calculation(arguments, parameters)


class UnVectorizedTableAlgorithm(Algorithm):
    @staticmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, (pd.Series, np.ndarray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters])
        return calculation(arguments, parameters)



