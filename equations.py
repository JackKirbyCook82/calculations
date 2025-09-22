# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Equation Objects
@author: Jack Kirby Cook

"""

import inspect
from itertools import chain
from functools import reduce, wraps
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.meta import ProxyMeta, AttributeMeta
from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variable", "Equation"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class VariableMeta(ProxyMeta, AttributeMeta): pass
class Variable(Node, ABC, metaclass=VariableMeta):
    def __bool__(self): return bool(self.varvalue is not None)
    def __init__(self, varkey, varname, vartype, *args, **kwargs):
        super().__init__(*args, linear=False, multiple=False, **kwargs)
        self.__vartype = vartype
        self.__varname = varname
        self.__varkey = varkey
        self.__varvalue = None

    @abstractmethod
    def calculation(self, order): pass

    @property
    def sources(self):
        children = self.children.values()
        if not bool(self): generator = (variable for child in children for variable in child.sources)
        else: generator = iter([self])
        yield from generator

    @property
    def vartype(self): return self.__vartype
    @property
    def varname(self): return self.__varname
    @property
    def varkey(self): return self.__varkey
    @property
    def varvalue(self): return self.__varvalue
    @varvalue.setter
    def varvalue(self, value): self.__varvalue = value


class SourceVariable(Variable, ABC):
    def __init__(self, *args, locator, **kwargs):
        super().__init__(*args, **kwargs)
        self.__locator = locator

    @property
    def locator(self): return self.__locator


class DerivedVariable(Variable, ABC, attribute="Derived"):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = inspect.signature(function).parameters.items()
        arguments = [key for key, value in signature if value.kind != value.KEYWORD_ONLY]
        parameters = [key for key, value in signature if value.kind == value.KEYWORD_ONLY]
        domain = Domain(arguments, parameters)
        self.__function = function
        self.__domain = domain

    def calculation(self, order):
        children = list(self.children.items())
        if bool(self): wrapper = lambda arguments, parameters: self.value
        else:
            primary = [variable.calculation(order) for key, variable in children if key in self.domain.arguments]
            secondary = {key: variable.calculation(order) for key, variable in children if key in self.domain.parameters}
            calculations = Domain(primary, secondary)
            primary = lambda arguments, parameters: [calculation(arguments, parameters) for calculation in calculations.arguments]
            secondary = lambda arguments, parameters: {key: calculation(arguments, parameters) for key, calculation in calculations.parameters.items()}
            wrapper = lambda arguments, parameters: self.function(*primary(arguments, parameters), **secondary(arguments, parameters))
        return wrapper

    @property
    def function(self): return self.__function
    @property
    def domain(self): return self.__domain


class ArgumentVariable(SourceVariable, ABC, attribute="Argument"):
    def calculation(self, order):
        argument = order.index(self)
        wrapper = lambda arguments, parameters: arguments[argument]
        return wrapper

class ParameterVariable(SourceVariable, ABC, attribute="Parameter"):
    def calculation(self, order):
        parameter = str(self.varkey)
        wrapper = lambda arguments, parameters: parameters[parameter]
        return wrapper


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

    def __add__(cls, others):
        assert isinstance(others, list) or issubclass(others, Equation)
        assert all([issubclass(other, Equation) for other in others]) if isinstance(others, list) else True
        function = lambda string: str(string).replace("Equation", "")
        others = others if isinstance(others, list) else [others]
        name = "".join([function(other.__name__) for other in others])
        bases = reversed([cls] + others)
        equation = EquationMeta(str(name), tuple(bases), dict())
        return equation

    def __call__(cls, sources, *args, **kwargs):
        variables = [proxy(initialize=True) for key, proxy in cls.proxys.items()]
        assert all([isinstance(variable, Variable) for variable in variables])
        variables = {str(variable.varkey): variable for variable in variables}
        variables = cls.connect(variables)
        variables = cls.source(variables, sources=sources)
        return super(EquationMeta, cls).__call__(variables, *args, **kwargs)

    @staticmethod
    def connect(variables):
        assert isinstance(variables, dict)
        for variable in variables.values():
            if not isinstance(variable, DerivedVariable): continue
            for key in list(variable.domain):
                variable[key] = variables[key]
        return variables

    @staticmethod
    def source(variables, *, sources):
        assert isinstance(variables, dict)
        for variable in variables.values():
            name = str(variable.locator)
            if isinstance(variable, DerivedVariable): continue
            else: variable.value = sources.get(name, None)
        return variables

    @property
    def proxys(cls): return cls.__proxys__


class Equation(ABC, metaclass=EquationMeta):
    def __init_subclass__(cls, *args, **kwargs): pass
    def __init__(self, variables, *args, **kwargs):
        assert isinstance(variables, dict)
        assert all([isinstance(variable, Variable) for variable in variables.values()])
        self.__variables = dict(variables)

    def __getattr__(self, attribute):
        variables = {key: variable for key, variable in self.variables.items()}
        if attribute not in variables.keys():
            raise AttributeError(attribute)
        variable = variables[attribute]
        calculation = self.calculation(variable)
        return calculation

    def calculation(self, variable):
        sources = list(set(variable.sources))
        arguments = ODict([(source, source.content) for source in sources if isinstance(source, ArgumentVariable)])
        parameters = ODict([(source, source.content) for source in sources if isinstance(source, ParameterVariable)])
        parameters = {str(variable.varkey): content for variable, content in parameters.items()}
        order = list(arguments.keys())
        arguments = list(arguments.values())
        calculation = variable.calculation(order)

        @wraps(self.calculation)
        def wrapper(*args, **kwargs):
            value = self.algorithm(calculation, arguments, parameters, *args, vartype=variable.vartype, **kwargs)
            value = value.astype(variable.vartype)
            variable.varvalue = value
            return value
        return wrapper

    @abstractmethod
    def algorithm(self, calculation, arguments, parameters, *args, **kwargs): pass
    @property
    def variables(self): return self.__variables



