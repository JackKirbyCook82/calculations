# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import os
import sys
from abc import ABC

MODULE = os.path.dirname(os.path.realpath(__file__))
if MODULE not in sys.path: sys.path.append(MODULE)

from algorithms import NumericAlgorithm, IntegralAlgorithm, VectorAlgorithm, ArrayAlgorithm, TableAlgorithm
from computations import ArrayComputation, TableComputation
from equations import Equation, Variable
from support.concepts import Assembly

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Equations"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class ConstantVariable(Variable.Parameter):
    pass


class IndependentVariable(Variable.Argument):
    pass


class DependentVariable(Variable.Derived):
    pass


class NumericEquation(NumericAlgorithm, Equation, ABC):
    pass


class IntegralEquation(IntegralAlgorithm, Equation, ABC):
    pass


class VectorEquation(ArrayComputation, VectorAlgorithm, Equation, ABC):
    pass


class ArrayEquation(ArrayComputation, ArrayAlgorithm, Equation, ABC):
    pass


class TableEquation(TableComputation, TableAlgorithm, Equation, ABC):
    pass


class Equations(Assembly): Numeric, Integral, Vector, Array, Table = NumericEquation, IntegralEquation, VectorEquation, ArrayEquation, TableEquation
class Variables(Assembly): Constant, Independent, Dependent = ConstantVariable, IndependentVariable, DependentVariable



