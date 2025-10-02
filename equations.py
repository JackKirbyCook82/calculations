# -*- coding: utf-8 -*-
"""
Created on Tues Aug 12 2025
@name:   Equation Objects
@author: Jack Kirby Cook

"""

import inspect
import regex as re
from enum import StrEnum
from itertools import chain
from functools import reduce, wraps
from abc import ABC, ABCMeta, abstractmethod
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.trees import Node

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Equation", "DependentVariable", "IndependentVariable", "ConstantVariable", "DomainError"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


VariableTyping = StrEnum("VariableTyping", ["CONSTANT", "INDEPENDENT", "DEPENDENT"])
class Proxy(object):
    def __init__(self, variable, arguments, parameters):
        assert isinstance(arguments, tuple) and isinstance(parameters, dict)
        self.__parameters = parameters
        self.__arguments = arguments
        self.__variable = variable

    def __call__(self, *args, **kwargs):
        arguments = self.arguments + tuple(args)
        parameters = self.parameters | dict(kwargs)
        instance = self.variable(*arguments, **parameters)
        return instance

    @property
    def vartyping(self): return self.variable.vartyping

    @property
    def parameters(self): return self.__parameters
    @property
    def arguments(self): return self.__arguments
    @property
    def variable(self): return self.__variable


class Deferred(object):
    def __init__(self, variable): self.variable = variable
    def __call__(self, *arguments, **parameters):
        return Proxy(self.variable, arguments, parameters)

    @property
    def vartyping(self): return self.variable.vartyping


class DomainError(Exception): pass
class Domain(ntuple("Domain", "arguments parameters")):
    def __iter__(self): return chain(self.arguments, self.parameters)


class Variable(Node, ABC):
    def __init_subclass__(cls, *args, **kwargs):
        cls.vartyping = kwargs.get("vartyping", getattr(cls, "vartyping", None))

    def __bool__(self): return bool(self.varvalue is not None)
    def __str__(self): return self.render()
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
        if bool(self): generator = iter([self])
        elif bool(self.terminal): generator = iter([self])
        else: generator = (source for child in children for source in child.sources)
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
        locator = locator if isinstance(locator, tuple) else tuple([locator])
        self.__locator = locator

    @property
    def locator(self): return self.__locator


@Deferred
class DependentVariable(Variable, ABC, vartyping=VariableTyping.DEPENDENT):
    def __init__(self, *args, function, **kwargs):
        super().__init__(*args, **kwargs)
        signature = list(inspect.signature(function).parameters.values())
        arguments = [str(value) for value in signature if value.kind == value.POSITIONAL_OR_KEYWORD]
        parameters = [str(value) for value in signature if value.kind == value.KEYWORD_ONLY]
        domain = Domain(arguments, parameters)
        self.__function = function
        self.__domain = domain

    def calculation(self, order):
        children = list(self.children.items())
        if bool(self): wrapper = lambda arguments, parameters: self.varvalue
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


@Deferred
class IndependentVariable(SourceVariable, ABC, vartyping=VariableTyping.INDEPENDENT):
    def calculation(self, order):
        argument = order.index(self)
        wrapper = lambda arguments, parameters: arguments[argument]
        return wrapper

@Deferred
class ConstantVariable(SourceVariable, ABC, vartyping=VariableTyping.CONSTANT):
    def calculation(self, order):
        parameter = str(self.varkey)
        wrapper = lambda arguments, parameters: parameters[parameter]
        return wrapper


class EquationMeta(ABCMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        exclude = [key for key, value in attrs.items() if isinstance(value, Proxy)]
        attrs = {key: value for key, value in attrs.items() if key not in exclude}
        cls = super(EquationMeta, mcs).__new__(mcs, name, bases, attrs, *args, **kwargs)
        return cls

    def __init__(cls, name, bases, attrs, *args, **kwargs):
        super(EquationMeta, cls).__init__(name, bases, attrs, *args, **kwargs)
        existing = [dict(base.proxys) for base in bases if issubclass(type(base), EquationMeta)]
        existing = reduce(lambda lead, lag: lead | lag, existing, dict())
        updated = {key: value for key, value in attrs.items() if isinstance(value, Proxy)}
        cls.__proxys__ = dict(existing) | dict(updated)

    def __add__(cls, others):
        assert isinstance(others, list) or issubclass(others, Equation)
        assert all([issubclass(other, Equation) for other in others]) if isinstance(others, list) else True
        split = lambda string: re.findall(r'[A-Z][a-z]*', str(string).replace("Equation", ""))
        bases = (others if isinstance(others, list) else [others]) + [cls]
        names = list(chain(*[split(base.__name__) for base in bases]))
        name = "".join(list(dict.fromkeys(names))) + "Equation"
        equation = EquationMeta(str(name), tuple(bases), dict())
        return equation

    def __call__(cls, *args, arguments, parameters, **kwargs):
        variables = [proxy(*args, **kwargs) for key, proxy in cls.proxys.items()]
        assert all([isinstance(variable, Variable) for variable in variables])
        variables = {str(variable.varkey): variable for variable in variables}
        variables = cls.connect(variables)
        variables = cls.populate(variables, arguments, parameters)
        return super(EquationMeta, cls).__call__(variables, *args, **kwargs)

    @staticmethod
    def connect(variables):
        assert isinstance(variables, dict)
        for variable in variables.values():
            try: domain = list(variable.domain)
            except AttributeError: continue
            for key in domain: variable[key] = variables[key]
        return variables

    @staticmethod
    def populate(variables, arguments, parameters):
        locate = lambda sources, locator: sources.get(locator, {})
        function = lambda sources, locator, *locators: function(locate(sources, locator), *locators) if bool(locators) else locate(sources, locator)
        assert isinstance(variables, dict)
        for variable in variables.values():
            if variable.vartyping is VariableTyping.DEPENDENT: continue
            elif variable.vartyping is VariableTyping.INDEPENDENT: value = function(arguments, *variable.locator)
            elif variable.vartyping is VariableTyping.CONSTANT: value = function(parameters, *variable.locator)
            else: raise ValueError(variable.vartyping)
            variable.varvalue = value
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
        if variable.terminal: return lambda: (variable.varname, variable.varvalue)
        calculation = self.calculation(variable)
        return calculation

    def __call__(self, *args, **kwargs):
        generator = self.execute(*args, **kwargs)
        contents = dict(generator)
        content = self.computation(contents, *args, **kwargs)
        return content

    def calculation(self, variable):
        sources = list(set(variable.sources))
        if not all([bool(source) for source in sources]): raise DomainError()
        arguments = ODict([(source, source.varvalue) for source in sources if source.vartyping is VariableTyping.INDEPENDENT])
        parameters = ODict([(source, source.varvalue) for source in sources if source.vartyping is VariableTyping.CONSTANT])
        parameters = {str(variable.varkey): value for variable, value in parameters.items()}
        order = list(arguments.keys())
        arguments = list(arguments.values())
        calculation = variable.calculation(order)

        @wraps(self.calculation)
        def wrapper(*args, **kwargs):
            value = self.algorithm(calculation, arguments, parameters, *args, vartype=variable.vartype, **kwargs)
            value = value.astype(variable.vartype)
            variable.varvalue = value
            return variable.varname, value
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



