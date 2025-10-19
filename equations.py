# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Equation Objects
@author: Jack Kirby Cook

"""

import inspect
import regex as re
import pandas as pd
import xarray as xr
from enum import StrEnum
from itertools import chain
from functools import reduce, wraps
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import AttributeMeta
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "DependentVariable", "IndependentVariable", "ConstantVariable", "VariableError"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class VariableError(Exception, metaclass=AttributeMeta): pass
class VariableDomainError(VariableError, attribute="Domain"): pass


class Variable(Node, ABC):
    def __bool__(self): return bool(self.varvalue is not None)
    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(varkey, *args, **kwargs)
        self.__vartype = vartype
        self.__varname = varname
        self.__varkey = varkey

    @abstractmethod
    def calculation(self, order): pass

    @property
    def sources(self):
        children = list(self.children.values())
        if bool(children): generator = iter([self])
        else: generator = (source for child in children for source in child.sources)
        yield from generator

    @property
    def vartype(self): return self.__vartype
    @property
    def varname(self): return self.__varname
    @property
    def varkey(self): return self.__varkey


class SourceVariable(Variable, ABC):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, **kwargs)
        self.__locator = locator if isinstance(locator, tuple) else tuple([locator])

    def calculation(self, order):
        @wraps(self.calculation)
        def wrapper(arguments, parameters):
            pass
        return wrapper

    @property
    def locator(self): return self.__locator


# class ConstantVariable(SourceVariable, ABC):
#    def calculation(self, order):
#        wrapper = lambda arguments, parameters: parameters[str(self.varkey)]
#        return wrapper

# class IndependentVariable(SourceVariable, ABC):
#    def calculation(self, order):
#        wrapper = lambda arguments, parameters: arguments[order.index(self)]
#        return wrapper

# class DependentVariable(DerivedVariable, ABC):
#     pass

#    @Dispatchers.Type(locator=0)
#    def locate(cls, sources, locator, *locators): raise TypeError(type(sources))
#    @locate.register(xr.Dataset, pd.DataFrame)
#    def table(cls, sources, locator, *locators): return sources.get(locator, None)
#    @locate.register(dict)
#    def mapping(cls, sources, locator, *locators): return cls.locate(sources.get(locator, None), *locators)
#    @locate.register(types.NoneType)
#    def empty(cls, sources, locator, *locators): return None


class DerivedVariable(Variable, ABC):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = list(inspect.signature(function).parameters.values())
        arguments = [str(value) for value in signature if value.kind == value.POSITIONAL_OR_KEYWORD]
        parameters = [str(value) for value in signature if value.kind == value.KEYWORD_ONLY]
        self.__domain = Domain(arguments, parameters)
        self.__function = function

    def calculation(self, order):
        variables = list(self.children.items())
        primary = [variable.calculation(order) for key, variable in variables if key in self.domain.arguments]
        secondary = {key: variable.calculation(order) for key, variable in variables if key in self.domain.parameters}
        calculations = Domain(primary, secondary)

        @wraps(self.calculation)
        def wrapper(arguments, parameters):
            arguments = [calculation(arguments, parameters) for calculation in calculations.arguments]
            parameters = {key: calculation(arguments, parameters) for key, calculation in calculations.parameters.items()}
            return self.function(*arguments, **parameters)
        return wrapper

    @property
    def function(self): return self.__function
    @property
    def domain(self): return self.__domain


class Factor(ABC):
    def __init_subclass__(cls, *args, variable, **kwargs):
        cls.__variable__ = variable

    def __init__(self, *arguments, **parameters):
        assert isinstance(arguments, tuple) and isinstance(parameters, dict)
        self.__parameters = parameters
        self.__arguments = arguments

    def __call__(self, *args, **kwargs):
        arguments = tuple(self.arguments) + tuple(args)
        parameters = dict(self.parameters) | dict(kwargs)
        instance = self.variable(*arguments, **parameters)
        return instance

    @property
    def variable(self): return type(self).__variable__
    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments


class EquationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Factor)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = [dict(base.proxys) for base in bases if issubclass(type(base), EquationMeta)]
        existing = reduce(lambda lead, lag: lead | lag, existing, dict())
        updated = {key: value for key, value in attrs.items() if isinstance(value, Factor)}
        cls.__factors__ = dict(existing) | dict(updated)

    def __add__(cls, others):
        assert isinstance(others, list) or issubclass(others, Equation)
        assert all([issubclass(other, Equation) for other in others]) if isinstance(others, list) else True
        split = lambda string: re.findall(r'[A-Z][a-z]*', str(string).replace("Equation", ""))
        bases = (others if isinstance(others, list) else [others]) + [cls]
        names = list(chain(*[split(base.__name__) for base in bases]))
        name = "".join(list(dict.fromkeys(names))) + "Equation"
        equation = EquationMeta(str(name), tuple(bases), dict())
        return equation

    def __call__(cls, *args, **kwargs):
        variables = [factor(*args, **kwargs) for key, factor in cls.factors.items()]
        assert all([isinstance(variable, Variable) for variable in variables])
        variables = {str(variable.varkey): variable for variable in variables}
        for variable in variables.values():
            for key in list(variable.domain): variable[key] = variables[key]
        return super(EquationMeta, cls).__call__(variables, *args, **kwargs)

    @property
    def factors(cls): return cls.__factors__


class Equation(ABC, metaclass=EquationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, variables, *args, **kwargs):
        assert isinstance(variables, dict)
        assert all([isinstance(variable, Variable) for variable in variables.values()])
        self.__variables = dict(variables)

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys(): raise AttributeError(attribute)
        variable = variables[attribute]
        calculation = self.calculation(variable)
        return calculation

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        contents = dict(generator)
        content = self.computation(contents, *args, **kwargs)
        return content

    def calculation(self, variable):
        order = list(set(variable.sources))
        calculation = variable.calculation(order)

        @wraps(self.calculation)
        def wrapper(*arguments, **parameters):
            results = self.algorithm(calculation, arguments, parameters, vartype=variable.vartype)
            return results.astype(variable.vartype)
        return wrapper

    @staticmethod
    @abstractmethod
    def algorithm(calculation, arguments, parameters, *args, **kwargs): pass
    @staticmethod
    @abstractmethod
    def computation(contents, *args, **kwargs): pass

    @staticmethod
    def execute(*args, **kwargs): return; yield

    @property
    def variables(self): return self.__variables



