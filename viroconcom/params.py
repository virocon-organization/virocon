#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter for distributions.
"""

from abc import ABC, abstractmethod

import numpy as np

import warnings # TODO remove warnings as soon as changes are done

__all__ = ["Param", "ConstantParam", "FunctionParam", "Wrapper"]


class Param(ABC):
    """
    Abstract base class for callable parameters.

    Replaces a constant or a function, so no distinction has to be made.
    """

    def __call__(self, x):
        """
        Parameters
        ----------
        x : float or array_like
            Point(s) at which to evaluate Param.

        Returns
        -------
        self._value(x) : float or list
            If x is an iterable a list of the same length will be returned,
            else if x is a scalar a float will be returned.

        """
        try:
            return [self._value(y) for y in x]
        except TypeError:  # not iterable
            return self._value(x)

    @abstractmethod
    def _value(self, x):
        """
        The value at x. Will be returned on call.
        """
        pass


class ConstantParam(Param):
    """A constant, but callable parameter."""

    def __init__(self, constant):
        """
        Parameters
        ----------
        constant : scalar
            The constant value to return.
        """
        self._constant = float(constant)

    def _value(self, _):
        return self._constant

    def __str__(self):
        return str(self._constant)



class FunctionParam(Param):
    """A callable parameter, which depends on the value supplied."""

    def __init__(self, a, b, c, func_type, wrapper=None):
        """
        Parameters
        ----------
        a,b,c : float
            The function parameters.
        func_type : str
            Defines which kind of dependence function to use:
                :power3: :math:`a + b * x^c`
                :exp3: :math:`a + b * e^{x * c}`
        wrapper : function or Wrapper
            A function or a Wrapper object to wrap around the function.
            The function has to be pickleable. (i.e. lambdas, clojures, etc. are not supported.)
            Using this wrapper, one can e.g. create :math:`exp(a + b * x^c)`
            with func_type=polynomial and wrapper=math.exp.
        """

        self.a = a
        self.b = b
        self.c = c

        if func_type == "power3":
            self._func = self._power3
            self.func_name = "power3"
        elif func_type == "exp3":
            self._func = self._exp3
            self.func_name = "exp3"
        else:
            raise ValueError("{} is not a known kind of function.".format(func_type))

        if wrapper is None:
            self._wrapper = Wrapper(self._identity)
        elif isinstance(wrapper, Wrapper):
            self._wrapper = wrapper
        elif callable(wrapper):
            self._wrapper = Wrapper(wrapper)
        else:
            raise ValueError("Wrapper has to be a callable.")
        # TODO test wrapper with Wrapper and with function object

    def _identity(self, x):
        return x

    # The 3-parameter power function (a dependence function).
    def _power3(self, x):
        return self.a + self.b * x ** self.c

    # The 3-parameter exponential function (a dependence function).
    def _exp3(self, x):
        return self.a + self.b * np.exp(self.c * x)

    def _value(self, x):
        return self._wrapper(self._func(x))

    def __str__(self):
        if self.func_name == "power3":
            function_string = "" + str(self.a) + "+" + str(self.b) + "x" + "^{" + str(self.c) + "}"
        elif self.func_name == "exp3":
            function_string = "" + str(self.a) + "+" + str(self.b) + "e^{" + str(self.c) + "x}"
        if isinstance(self._wrapper.func, np.ufunc):
            function_string += " with _wrapper: " + str(self._wrapper)
        return function_string


class Wrapper():

    def __init__(self, func, inner_wrapper=None):
        self.func = func
        self.inner_wrapper = inner_wrapper

    def __call__(self, x):
        if self.inner_wrapper is None:
            return self.func(x)
        else:
            return self.func(self.inner_wrapper(x))

    def __str__(self):
        return "Wrapper with function '" + str(self.func) + "' and inner_wrapper '" + str(self.inner_wrapper) + '"'
