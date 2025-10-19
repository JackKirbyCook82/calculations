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

from algorithms import NumericAlgorithm, VectorizedArrayAlgorithm, VectorizedTableAlgorithm, UnVectorizedArrayAlgorithm, UnVectorizedTableAlgorithm
from equations import Equation, ConstantVariable, IndependentVariable, DependentVariable, VariableError
from computations import ArrayComputation, TableComputation
from support.concepts import Assembly

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Equations", "Errors"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class VectorizedArrayEquation(ArrayComputation, VectorizedArrayAlgorithm, Equation, ABC): pass
class VectorizedTableEquation(TableComputation, VectorizedTableAlgorithm, Equation, ABC): pass
class UnVectorizedArrayEquation(ArrayComputation, UnVectorizedArrayAlgorithm, Equation, ABC): pass
class UnVectorizedTableEquation(TableComputation, UnVectorizedTableAlgorithm, Equation, ABC): pass
class NumericEquation(NumericAlgorithm, Equation, ABC): pass

class Variables(Assembly): Constant, Independent, Dependent = ConstantVariable, IndependentVariable, DependentVariable
class Errors(Assembly): Domain = VariableError.Domain
class Equations(Assembly):
    class UnVectorized(Assembly): Array, Table = UnVectorizedArrayEquation, UnVectorizedTableEquation
    class Vectorized(Assembly): Array, Table = VectorizedArrayEquation, VectorizedTableEquation
    class Calculus(Assembly): Integral = IntegralEquation
    class Algebra(Assembly): Numeric = NumericEquation




