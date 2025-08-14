# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 12 2025
@name:   Variable Objects
@author: Jack Kirby Cook

"""

import inspect
from itertools import chain
from functools import reduce
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple

from support.meta import ProxyMeta
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class Variable(Node, ABC, metaclass=ProxyMeta):
    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(*args, linear=False, multiple=False, **kwargs)
        self.__name = varname
        self.__type = vartype
        self.__key = varkey
        self.__value = None

    def __bool__(self): return self.value is not None
    def __repr__(self): return str(self.key)
    def __str__(self): return str(self.name)

    @abstractmethod
    def execute(self, order): pass

    @property
    def sources(self):
        children = self.children.values()
        if not bool(self): generator = (variable for child in children for variable in child.sources)
        else: generator = iter([self])
        yield from generator

    @property
    def name(self): return self.__name
    @property
    def type(self): return self.__type
    @property
    def key(self): return self.__key
    @property
    def value(self): return self.__value
    @value.setter
    def value(self, value): self.__value = value


class ArgumentVariable(Variable, ABC):
    def execute(self, order):
        argument = order.index(self)
        wrapper = lambda arguments, parameters: arguments[argument]
        wrapper.__name__ = repr(self)
        return wrapper


class ParameterVariable(Variable, ABC):
    def execute(self, order):
        parameter = str(self)
        wrapper = lambda arguments, parameters: parameters[parameter]
        wrapper.__name__ = repr(self)
        return wrapper


class DerivedVariable(Variable, ABC):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = inspect.signature(function).parameters.items()
        arguments = [key for key, value in signature if value.kind != value.KEYWORD_ONLY]
        parameters = [key for key, value in signature if value.kind == value.KEYWORD_ONLY]
        domain = Domain(arguments, parameters)
        self.__function = function
        self.__domain = domain

    def execute(self, order):
        children = list(self.children.items())
        if bool(self): wrapper = lambda arguments, parameters: self.varValue
        else:
            primary = [variable.execute(order) for key, variable in children if key in self.domain.arguments]
            secondary = {key: variable.execute(order) for key, variable in children if key in self.domain.parameters}
            executes = Domain(primary, secondary)
            primary = lambda arguments, parameters: [execute(arguments, parameters) for execute in executes.arguments]
            secondary = lambda arguments, parameters: {key: execute(arguments, parameters) for key, execute in executes.parameters.items()}
            wrapper = lambda arguments, parameters: self.function(*primary(arguments, parameters), **secondary(arguments, parameters))
        wrapper.__name__ = repr(self)
        return wrapper

    @property
    def function(self): return self.__function
    @property
    def domain(self): return self.__domain


class EquationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, proxy in attrs.items() if isinstance(proxy, Variable)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = [dict(base.proxys) for base in bases if issubclass(type(base), EquationMeta)]
        existing = reduce(lambda lead, lag: lead | lag, existing, dict())
        updated = {key: proxy for key, proxy in attrs.items() if issubclass(proxy, Variable)}
        cls.__proxys__ = dict(existing) | dict(updated)

    def __call__(cls, *args, **kwargs):
        variables = [proxy(*args, **kwargs) for key, proxy in cls.proxys.items()]
        assert all([isinstance(variable, Variable) for variable in variables])
        variables = {repr(variable): variable for variable in variables}
        for variable in variables.values():
            if not isinstance(variable, DerivedVariable): continue
            for key in list(variable.domain):
                variable[key] = variables[key]
        return super(EquationMeta, cls).__call__(variables, *args, **kwargs)

    @property
    def proxys(cls): return cls.__proxys__


class Equation(ABC, metaclass=EquationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, variables, *args, **kwargs):
        assert isinstance(variables, dict)
        assert all([isinstance(variable, Variable) for variable in variables.values()])
        self.__variables = dict(variables)

    def __enter__(self): return self
    def __exit__(self, error_type, error_value, error_traceback):
        for key in list(self.variables.keys()):
            del self.variables[key]

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        return variables[attribute]

    @property
    def variables(self): return self.__variables

