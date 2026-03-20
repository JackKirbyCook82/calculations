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

from equations import Equation, Variables, Errors
from computations import Computations
from algorithms import Algorithms

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "Variables", "Computations", "Algorithms", "Errors"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"



