# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Equation Objects
@author: Jack Kirby Cook

"""

import types
import inspect
import regex as re
import pandas as pd
import xarray as xr
from enum import Enum
from itertools import chain
from functools import reduce, wraps
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import RegistryMeta, AttributeMeta
from support.decorators import Dispatchers
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "Factor", "Error", "Variable"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class VariableType(Enum): DEPENDENT, INDEPENDENT, CONSTANT, SOURCE = list(range(4))
class Variable(Node, ABC, metaclass=RegistryMeta):
    Type = VariableType

    def __repr__(self): return f"{self.__class__.__name__}({self.varkey}, {self.varname})"
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
        if not bool(children): generator = iter([self])
        else: generator = (source for child in children for source in child.sources)
        yield from generator

    @property
    def vartype(self): return self.__vartype
    @property
    def varname(self): return self.__varname
    @property
    def varkey(self): return self.__varkey


class DependentVariable(Variable, ABC, register=VariableType.DEPENDENT):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = list(inspect.signature(function).parameters.values())
        arguments = [str(value) for value in signature if value.kind == value.POSITIONAL_OR_KEYWORD]
        parameters = [str(value) for value in signature if value.kind == value.KEYWORD_ONLY]
        self.__domain = Domain(arguments, parameters)
        self.__function = function

    def calculation(self, order):
        domain = list(self.children.items())
        primary = [variable.calculation(order) for key, variable in domain if key in self.domain.arguments]
        secondary = {key: variable.calculation(order) for key, variable in domain if key in self.domain.parameters}
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


class SourceVariable(Variable, ABC, register=VariableType.SOURCE):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, **kwargs)
        self.__locator = locator

    def calculation(self, order):
        @wraps(self.calculation)
        def wrapper(arguments, parameters):
            if isinstance(self, IndependentVariable): return arguments[int(order.index(self))]
            elif isinstance(self, ConstantVariable): return parameters[str(self)]
            else: raise Error[VariableType.DEPENDENT]()
        return wrapper

    @abstractmethod
    def locate(self, *args, **kwargs): pass
    @property
    def locator(self): return self.__locator


class IndependentVariable(SourceVariable, ABC, register=VariableType.INDEPENDENT):
    def __init__(self, *args, locator, **kwargs):
        locator = locator if isinstance(locator, tuple) else tuple([locator])
        super().__init__(*args, locator=locator, **kwargs)

    def locate(self, arguments):
        locator = list(self.locator)
        content = self.find(arguments, *locator)
        if isinstance(content, types.NoneType): raise Error.Independent()
        return content

    @Dispatchers.Type(locator=0)
    def find(self, arguments, locator, *locators): raise TypeError(type(arguments))
    @find.register(xr.Dataset, pd.DataFrame)
    def table(self, arguments, locator, *locators): return arguments.get(locator, None)
    @find.register(dict, list)
    def mapping(self, arguments, locator, *locators): return self.find(arguments.get(locator, None), *locators)
    @find.register(types.NoneType)
    def empty(self, arguments, locator, *locators): return None


class ConstantVariable(SourceVariable, ABC, register=VariableType.CONSTANT):
    def __init__(self, *args, locator, **kwargs):
        assert isinstance(locator, str)
        super().__init__(*args, locator=locator, **kwargs)

    def locate(self, parameters):
        assert isinstance(parameters, dict)
        content = parameters.get(self.locator, None)
        if isinstance(content, types.NoneType): raise Error.Constant()
        return content


class Error(Exception, metaclass=AttributeMeta): pass
class DependentError(Error, attribute=str(VariableType.DEPENDENT.name).title()): pass
class IndependentError(Error, attribute=str(VariableType.INDEPENDENT.name).title()): pass
class ConstantError(Error, attribute=str(VariableType.CONSTANT.name).title()): pass
class SourceError(Error, attribute=str(VariableType.SOURCE.name).title()): pass


class Factor(object, metaclass=RegistryMeta):
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


class DependentFactor(Factor, variable=DependentVariable, register=VariableType.DEPENDENT): pass
class IndependentFactor(Factor, variable=IndependentVariable, register=VariableType.INDEPENDENT): pass
class ConstantFactor(Factor, variable=ConstantVariable, register=VariableType.CONSTANT): pass


class EquationMeta(RegistryMeta, ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Factor)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = [dict(base.factors) for base in bases if issubclass(type(base), EquationMeta)]
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
            if not isinstance(variable, SourceVariable):
                for key in list(variable.domain):
                    variable[key] = variables[key]
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
        if variable.terminal: function = self.retrieve(variable)
        else: function = self.calculation(variable)
        wrapper = lambda arguments, **parameters: (variable.varname, function(arguments, **parameters))
        return wrapper

    def __call__(self, arguments, **parameters):
        assert isinstance(arguments, (list, dict, pd.DataFrame, xr.Dataset))
        generator = self.execute(arguments, **parameters)
        contents = dict(generator)
        content = self.computation(contents)
        return content

    def retrieve(self, variable):
        children = list(set(variable.children))
        assert not bool(children)

        @wraps(self.retrieve)
        def wrapper(arguments, **parameters):
            if isinstance(variable, IndependentVariable): return variable.locate(arguments)
            elif isinstance(variable, ConstantVariable): return variable.locate(parameters)
            else: raise Error.Dependent()
        return wrapper

    def calculation(self, variable):
        sources = list(set(variable.sources))
        assert all([isinstance(source, SourceVariable) for source in sources])
        independents = list(filter(lambda source: isinstance(source, IndependentVariable), sources))
        constants = list(filter(lambda source: isinstance(source, ConstantVariable), sources))

        @wraps(self.calculation)
        def wrapper(arguments, **parameters):
            arguments = ODict([(source, source.locate(arguments)) for source in independents])
            parameters = ODict([(str(source), source.locate(parameters)) for source in constants])
            order, arguments = list(arguments.keys()), list(arguments.values())
            calculation = variable.calculation(order)
            content = self.algorithm(calculation, arguments, parameters, vartype=variable.vartype)
            return content.astype(variable.vartype)
        return wrapper

    @staticmethod
    @abstractmethod
    def algorithm(calculation, arguments, parameters, /, vartype): pass
    @staticmethod
    @abstractmethod
    def computation(contents): pass

    @staticmethod
    def execute(arguments, /, **parameters): return; yield

    @property
    def variables(self): return self.__variables


