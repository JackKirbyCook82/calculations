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

from equations import Equation, Variable, Factor, Error
from computations import Computation
from algorithms import Algorithm
from support.concepts import Assembly

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "Variables", "Computations", "Algorithms", "Errors"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Variables(Assembly):
    Dependent = Factor[Variable.Type.DEPENDENT]
    Independent = Factor[Variable.Type.INDEPENDENT]
    Constant = Factor[Variable.Type.CONSTANT]

class Computations(Assembly):
    Array = Computation[Computation.Type.Array]
    Table = Computation[Computation.Type.Table]

class Algorithms(Assembly):
    class Vectorized(Assembly):
        Array = Algorithm[Algorithm.Type.Array, True]
        Table = Algorithm[Algorithm.Type.Table, True]
    class UnVectorized(Assembly):
        Array = Algorithm[Algorithm.Type.Array, False]
        Table = Algorithm[Algorithm.Type.Table, False]
        Numeric = Algorithm[Algorithm.Type.Numeric, False]

class Errors(Assembly):
    Dependent = Error.Dependent
    Independent = Error.Independent
    Constant = Error.Constant
    Source = Error.Source



