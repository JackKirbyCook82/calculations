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
__all__ = ["NumericAlgorithm", "IntegralAlgorithm", "VectorAlgorithm", "ArrayAlgorithm", "TableAlgorithm"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Algorithm(ABC):
    @abstractmethod
    def execute(self, execute, arguments, parameters, *args, **kwargs): pass


class NumericAlgorithm(Algorithm):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, np.number) for argument in arguments])
        assert not any([isinstance(parameter, np.number) for parameter in parameters])
        return calculation(arguments, parameters)


class IntegralAlgorithm(Algorithm):
    def execute(self, execute, arguments, parameters, *args, **kwargs):
        pass


class VectorAlgorithm(Algorithm):
    def execute(self, calculation, arguments, parameters, *args, vartype, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        function = lambda *dataarrays, **constants: calculation(dataarrays, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[vartype], vectorize=True)


class ArrayAlgorithm(Algorithm):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, (xr.DataArray, np.ndarray, np.number)) for argument in arguments])
        assert not any([isinstance(parameter, xr.DataArray) for parameter in parameters])
        return calculation(arguments, parameters)


class TableAlgorithm(Algorithm):
    def execute(self, calculation, arguments, parameters, *args, **kwargs):
        assert all([isinstance(argument, pd.Series) for argument in arguments])
        assert not any([isinstance(parameter, pd.Series) for parameter in parameters])
        function = lambda dataframe, **constants: calculation(dataframe, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)

