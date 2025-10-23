# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 14 2025
@name:   Computation Objects
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
__all__ = ["Computation"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class ComputationType(Enum): Array, Table = list(range(1))
class Computation(ABC, metaclass=RegistryMeta):
    Type = ComputationType

    @staticmethod
    @abstractmethod
    def computation(contents): pass


class ArrayComputation(Computation, register=ComputationType.Array):
    @staticmethod
    def computation(contents):
        assert isinstance(contents, dict)
        assert all([isinstance(content, (xr.DataArray, np.number)) for content in contents.values()])
        dataarrays = {name: content for name, content in contents.items() if isinstance(content, xr.DataArray)}
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        datasets = [dataarray.to_dataset(name=name) for name, dataarray in dataarrays.items()]
        dataset = xr.merge(datasets)
        for name, content in numerics.items(): dataset[name] = content
        return dataset


class TableComputation(Computation, register=ComputationType.Table):
    @staticmethod
    def computation(contents):
        assert isinstance(contents, dict)
        assert all([isinstance(content, (pd.Series, np.number)) for content in contents.values()])
        series = [content.rename(name) for name, content in contents.items() if isinstance(content, pd.Series)]
        numerics = {name: content for name, content in contents.items() if isinstance(content, np.number)}
        dataframe = pd.concat(list(series), axis=1)
        for name, content in numerics.items(): dataframe[name] = content
        return dataframe



