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

from support.concepts import Assembly

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Algorithms"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class AlgorithmError(Exception): pass
class Algorithm(ABC):
    @staticmethod
    @abstractmethod
    def enforcement(arguments, parameters): pass
    @abstractmethod
    def algorithm(self, calculation, arguments, parameters, *args, **kwargs): pass


class TableEnforcement(ABC):
    @staticmethod
    def enforcement(arguments, parameters):
        if not all([isinstance(argument, (pd.Series, np.number)) for argument in arguments]):
            raise AlgorithmError([type(argument) for argument in arguments])
        if any([isinstance(parameter, (xr.Dataset, pd.Series)) for parameter in parameters.values()]):
            raise AlgorithmError({key: type(parameter) for key, parameter in parameters.items()})


class ArrayEnforcement(ABC):
    @staticmethod
    def enforcement(arguments, parameters):
        if not all([isinstance(argument, (xr.DataArray, np.number)) for argument in arguments]):
            raise AlgorithmError([type(argument) for argument in arguments])
        if any([isinstance(parameter, (xr.Dataset, pd.Series)) for parameter in parameters.values()]):
            raise AlgorithmError({key: type(parameter) for key, parameter in parameters.items()})


class VectorizedArrayAlgorithm(ArrayEnforcement, Algorithm):
    def algorithm(self, calculation, arguments, parameters, *args, vartype, **kwargs):
        self.enforcement(arguments, parameters)
        function = lambda *variables, **constants: calculation(variables, constants)
        return xr.apply_ufunc(function, *arguments, kwargs=parameters, output_dtypes=[vartype], vectorize=True)


class VectorizedTableAlgorithm(TableEnforcement, Algorithm):
    def algorithm(self, calculation, arguments, parameters, *args, **kwargs):
        self.enforcement(arguments, parameters)
        function = lambda variables, **constants: calculation(variables, constants)
        return pd.concat(arguments, axis=1).apply(function, axis=1, raw=True, **parameters)


class UnVectorizedAlgorithm(Algorithm, ABC):
    def algorithm(self, calculation, arguments, parameters, *args, **kwargs):
        self.enforcement(arguments, parameters)
        return calculation(arguments, parameters)


class UnVectorizedArrayAlgorithm(ArrayEnforcement, UnVectorizedAlgorithm): pass
class UnVectorizedTableAlgorithm(TableEnforcement, UnVectorizedAlgorithm): pass


class Algorithms(Assembly):
    class Vectorized(Assembly):
        Array = VectorizedArrayAlgorithm
        Table = VectorizedTableAlgorithm
    class UnVectorized(Assembly):
        Array = UnVectorizedArrayAlgorithm
        Table = UnVectorizedTableAlgorithm


