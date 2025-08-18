# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Calculation Objects
@author: Jack Kirby Cook

"""

import os
import sys

MODULE = os.path.dirname(os.path.realpath(__file__))
if MODULE not in sys.path: sys.path.append(MODULE)

from algorithms import NumericAlgorithm, IntegralAlgorithm, VectorAlgorithm, ArrayAlgorithm, TableAlgorithm
from equations import Equation, Variable
from support.variables import Category

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equations", "Variables"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class ConstantVariable(Variable.Parameter):
    pass


class IndependentVariable(Variable.Argument):
    pass


class DependentVariable(Variable.Derived):
    pass


class NumericEquation(NumericAlgorithm, Equation):
    pass


class IntegralEquation(IntegralAlgorithm, Equation):
    pass


class VectorEquation(VectorAlgorithm, Equation):
    pass


class ArrayEquation(ArrayAlgorithm, Equation):
    pass


class TableEquation(TableAlgorithm, Equation):
    pass


class Equations(Category): Numeric, Integral, Vector, Array, Table = NumericEquation, IntegralEquation, VectorEquation, ArrayEquation, TableEquation
class Variables(Category): Constant, Independent, Dependent = ConstantVariable, IndependentVariable, DependentVariable


