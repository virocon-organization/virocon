#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uni- and multivariate distributions.
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as sts
import scipy.optimize

from .settings import SHAPE_STRING, LOCATION_STRING, SCALE_STRING, SHAPE2_STRING
from .params import FunctionParam, ConstantParam, Wrapper

__all__ = ["Distribution", "ParametricDistribution", "WeibullDistribution",
           "LognormalDistribution", "NormalDistribution", "KernelDensityDistribution",
           "MultivariateDistribution"]


class Distribution(ABC):
    """
    Abstract base class for distributions.

    """

    @abstractmethod
    def cdf(self, x, rv_values, dependency):
        """Calculate the cumulative distribution function."""

    @abstractmethod
    def i_cdf(self, probabilities, rv_values, dependency):
        """Calculate percent-point function. (inverse cumulative distribution function)"""


class ParametricDistribution(Distribution, ABC):
    """
    Abstract base class for parametric distributions.

    Attributes
    ----------
    shape : Param
        The shape parameter.
    loc : Param
        The location parameter.
    scale : Param
        The scale parameter.
    shape2: Param
        Second shape parameter (if the distribution has one).
    name : str
        The name of the distribution. ("Weibull", "LogNormal", "Normal")
    _scipy_cdf : function
        The cumulative distribution function from scipy. (sts.weibull_min.cdf, ...)
    _scipy_i_cdf : function
        The inverse cumulative distribution (or percent-point) function.(sts.weibull_min.ppf, ...)
    _default_shape : float
        The default shape parameter.
    _default_loc : float
        The default loc parameter.
    _default_scale : float
        The default scale parameter.
    _default_shape2 : float
        Default shape2 parameter (if the distribution has a second shape parameter).


    Note
    -----
    The following attributes/methods need to be initialised by child classes:
        - name
        - _scipy_cdf
        - _scipy_i_cdf
    """

    @abstractmethod
    def __init__(self, shape, loc, scale, shape2=None):
        """
        Note
        ----
        All parametric distributions can be initalized with 'scale', 'shape'
        and 'loc' (location) parameters. Some distribution have a second shape
        parameter, 'shape2'. Implemented distributions are:

        * normal: :math:`f(x) = \\frac{1}{x\\widetilde{\\sigma} \\sqrt{2\\pi}}\\exp \\left[ \\frac{-(\\ln x - \\widetilde{\\mu})^2}{2\\widetilde{\\sigma}^2}\\right]`

        * Weibull: :math:`f(x) = \\frac{\\beta}{\\alpha}\\left( \\frac{x-\\gamma}{\\alpha}\\right)^{\\beta -1} \\exp \\left[-\\left( \\frac{x-\\gamma}{\\alpha} \\right)^{\\beta} \\right]`

        * log-normal: :math:`f(x) = \\frac{1}{x\\widetilde{\\sigma} \\sqrt{2\\pi}}\\exp \\left[ \\frac{-(\\ln x - \\widetilde{\\mu})^2}{2\\widetilde{\\sigma}^2}\\right]`

        * Exponentiated Weibull:

        Their scale, shape, and loc values corerspond to the variables
        in the probability density function in the following manner:

        ============  ===================  =================  ================  ================
        distribution  scale                shape              loc               shape2
        ============  ===================  =================  ================  ================
        normal        σ                    --                 μ                 --
        Weibull       α                    β                  γ                 --
        log-normal    e^μ                  σ                  --                --
        exp. Weibull  α                    β                  --                δ
        ============  ===================  =================  ================  ================

        Parameters
        ----------
        shape : Param,
            The shape parameter.
        loc : Param,
            The location parameter.
        scale : Param,
            The scale parameter.
        shape2: Param
            Second shape parameter (if the distribution has one).
        """
        self.shape = shape
        self.loc = loc
        self.scale = scale
        self.shape2 = shape2
        # The following attributes need to be overwritten by subclasses
        self.name = "Parametric"  # e.g. "Weibull", "Lognormal",  ...

        self._default_shape = 1
        self._default_loc = 0
        self._default_scale = 1
        self._default_shape2 = None

        self._valid_shape = {"min" : -np.inf, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True } # -inf < shape < inf
        self._valid_loc = {"min" : -np.inf, "strict_greater" : True,
                           "max" : np.inf, "strict_less" : True }
        self._valid_scale = {"min" : -np.inf, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }
        self._valid_shape2 = {"min" : -np.inf, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }


    @abstractmethod
    def _scipy_cdf(self, x, shape, loc, scale):
        """Overwrite with appropriate cdf function from scipy package. """

    @abstractmethod
    def _scipy_i_cdf(self, probabilities, shape, loc, scale):
        """Overwrite with appropriate i_cdf function from scipy package. """

    def cdf(self, x, rv_values=None, dependencies=None):
        """
        Calculate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to calculate the cdf.
        rv_values : array_like
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If x is an array, M must be len(x).
        dependencies : tuple
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.


        Returns
        -------
        cdf : ndarray,
            Cumulative distribution function evaluated at x under condition rv_values.
        """

        shape_val, loc_val, scale_val, shape2_val = self._get_parameter_values(rv_values, dependencies)
        if shape2_val == None:
            params = (shape_val, loc_val, scale_val)
        else:
            params = (shape_val, loc_val, scale_val, shape2_val)

        return self._scipy_cdf(x, *params)

    def i_cdf(self, probabilities, rv_values=None, dependencies=None):
        """
        Calculate percent-point function. (inverse cumulative distribution function)

        Parameters
        ----------
        probabilities : array_like
            Probabilities for which to calculate the i_cdf.
        rv_values : array_like
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If probabilities is an array, M must be len(probabilities).
        dependencies : tuple
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.

        Returns
        -------
        i_cdf : ndarray,
            Inverse cumulative distribution function evaluated for probabilities
            under condition rv_values.
        """

        shape_val, loc_val, scale_val, shape2_val = self._get_parameter_values(rv_values, dependencies)
        if shape2_val == None:
            params = (shape_val, loc_val, scale_val)
        else:
            params = (shape_val, loc_val, scale_val, shape2_val)

        return self._scipy_i_cdf(probabilities, *params)

    def ppf(self, probabilities, rv_values=None, dependencies=None):
        # Synsynom for i_cdf. Implemented that in external code a ParametricDistribution
        # object can be used when scipy.stats.rv_continuous.ppf is meant.
        return self.i_cdf(probabilities, rv_values, dependencies)

    def pdf(self, x, rv_values=None, dependencies=None):
        """
        Probability density function.

        Parameters
        ----------
        x : array_like
            Points at which the PDF should be evaluated.
        rv_values : array_like
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If x is an array, M must be len(x).
        dependencies : tuple
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.


        Returns
        -------
        cdf : ndarray,
            Porbability densities at x under condition rv_values.
        """

        shape_val, loc_val, scale_val, shape2_val = self._get_parameter_values(rv_values, dependencies)
        if shape2_val == None:
            params = (shape_val, loc_val, scale_val)
        else:
            params = (shape_val, loc_val, scale_val, shape2_val)

        return self._scipy_pdf(x, *params)

    def _get_parameter_values(self, rv_values, dependencies):
        """
        Evaluates the conditional shape, loc, scale parameters.

        Parameters
        ----------
        rv_values : array_like
            Values of all random variables in variable space in correct order.
        dependencies : tuple
            A 4-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.

        Returns
        -------
        parameter_vals : tuple
            A 4-element tuple with one entry each for the shape, loc and scale parameters.
            The tuple contains the values of the parameters evaluated under the conditions
            of dependencies.
            The values are either float or lists of float.
        """
        params = (self.shape, self.loc, self.scale, self.shape2)
        defaults = (self._default_shape, self._default_loc, self._default_scale, self._default_shape2)
        parameter_vals = []

        for i, param in enumerate(params):
            if param is None:
                parameter_vals.append(defaults[i])
            elif dependencies is None or dependencies[i] is None:
                parameter_vals.append(param(None))
                if parameter_vals[-1] is not None:
                    self._check_parameter_value(i, parameter_vals[-1])
            else:
                parameter_vals.append(param(rv_values[dependencies[i]]))
                try: # if list fo values iterate over values
                    for value in parameter_vals[-1]:
                        self._check_parameter_value(i, value)
                except TypeError:
                    self._check_parameter_value(i, parameter_vals[-1])

        return tuple(parameter_vals)

    def _check_parameter_value(self, param_index, param_value):
        """
        Checks if parameter values are within the distribution specific boundaries.

        Parameters
        ----------
        param_index : int
            Index of parameter.
            (0 = 'shape', 1 = 'loc', 2 = 'scale')
        param_value : float
            Value of parameter.

        Raises
        ------
        ValueError
            If parameter value is outside the boundaries.
        """

        if param_index == 0:
            valid = self._valid_shape
            param_name = SHAPE_STRING
        elif param_index == 1:
            valid = self._valid_loc
            param_name = LOCATION_STRING
        elif param_index == 2:
            valid = self._valid_scale
            param_name = SCALE_STRING
        elif param_index == 3:
            valid = self._valid_shape2
            param_name = SHAPE2_STRING

        if valid["strict_greater"]:
            if not param_value > valid["min"]:
                raise ValueError("Parameter out of bounds. {} has to be "
                                 "strictly greater than {}, but was {}"
                                 "".format(param_name, valid["min"], param_value))
        else:
            if not param_value >= valid["min"]:
                raise ValueError("Parameter out of bounds. {} has to be "
                                 "greater than {}, but was {}"
                                 "".format(param_name, valid["min"], param_value))

        if valid["strict_less"]:
            if not param_value < valid["max"]:
                raise ValueError("Parameter out of bounds. {} has to be "
                                 "strictly less than {}, but was {}"
                                 "".format(param_name, valid["max"], param_value))
        else:
            if not param_value <= valid["max"]:
                raise ValueError("Parameter out of bounds. {} has to be "
                                     "less than {}, but was {}"
                                     "".format(param_name, valid["max"], param_value))

        def __str__(self):
            return  "ParametricDistribution with shape={}, loc={}," \
                    "scale={}, shape2={}.".format(
                self.shape, self.loc, self.scale, self.shape2)

    @staticmethod
    def param_name_to_index(param_name):
        """
        Converts a parameter name ('shape', 'loc', 'scale') to the correct
        parameter index used in viroconcom (either 0, 1 or 2).

        Parameters
        ----------
        param_name : str
            The name of the parameter, must be 'shape', 'loc', or 'scale'.

        Returns
        -------
        param_index : int
            The index corresponding to the name of the parameter as it is
            internally defined in viroconcom.
        """
        param_index = None
        if param_name == SHAPE_STRING:
            param_index = 0
        elif param_name == LOCATION_STRING:
            param_index = 1
        elif param_name == SCALE_STRING:
            param_index = 2
        elif param_name == SHAPE2_STRING:
            param_index = 3
        else:
            raise ValueError("Wrong parameter name. The param_name variable "
                             "must be either 'shape', 'loc' or 'scale', however,"
                             "it was {}.".format(param_name))
        return param_index


class WeibullDistribution(ParametricDistribution):
    """
    A Weibull distribution.

    Examples
    --------
    Create a WeibullDistribution and plot the cumulative distribution function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from viroconcom.params import ConstantParam
    >>> shape = ConstantParam(1)
    >>> loc = ConstantParam(0)
    >>> scale = ConstantParam(1)
    >>> dist = WeibullDistribution(shape, loc, scale)
    >>> x = np.linspace(0, 5, num=100)
    >>> #file_example = plt.plot(x, dist.cdf(x, None, (None, None, None)),\
                                            #label="Weibull")
    >>> #legend = plt.legend()

    """

    def __init__(self, shape=None, loc=None, scale=None):
        super().__init__(shape, loc, scale)
        self.name = "Weibull"
        self._valid_shape = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }
        self._valid_scale = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }

    def _scipy_cdf(self, x, shape, loc, scale):
        return sts.weibull_min.cdf(x, c=shape, loc=loc, scale=scale)

    def _scipy_i_cdf(self, probabilities, shape, loc, scale):
        return sts.weibull_min.ppf(probabilities, c=shape, loc=loc, scale=scale)


class ExponentiatedWeibullDistribution(ParametricDistribution):
    """
    An exponentiated Weibull distribution.

    Note
    -----
    We use the parametrization that is also used in
    https://arxiv.org/pdf/1911.12835.pdf .
    """

    def __init__(self, shape=None, loc=None, scale=None, shape2=None):
        super().__init__(shape, loc, scale, shape2)
        self.name = "ExponentiatedWeibull"
        self._valid_shape = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }
        self._valid_scale = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }
        self._valid_shape2 = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }

    def _scipy_cdf(self, x, shape, loc, scale, shape2):
        x = np.array(x)
        # Ensure x> 0. In Matlab syntax do: x(x < 0) = 0
        indices = np.argwhere(x < 0)
        np.put(x, indices, np.zeros(indices.size))
        p = np.power(1 - np.exp(np.multiply(-1, np.power(np.divide(x,  scale), shape))), shape2)
        return p

    def _scipy_i_cdf(self, p, shape, loc, scale, shape2):
        p = np.array(p)
        if np.any(np.greater(p, 1)):
            p = np.nan
        if np.any(np.less(p, 0)):
            p = np.nan
        # In Matlab syntax: x = scale .* (-1 .* log(1 - p.^(1 ./ shape2))).^(1 ./ shape);
        x = np.multiply(scale, np.power(np.multiply(-1, np.log(1 - np.power(p, np.divide(1, shape2)))), np.divide(1, shape)))
        return x

    def _scipy_pdf(self, x, shape, loc, scale, shape2):
        """
        Probability density function of the exponentiated Weibull distribution.

        The parametrization from https://arxiv.org/pdf/1911.12835.pdf is used.

        Parameters
        ----------
        x : array_like
            Position where the PDF should be evaluated.
        shape: float
            beta in https://arxiv.org/pdf/1911.12835.pdf .
        loc : float
            The distribution does not have a location parameter.
        scale : float
            alpha in https://arxiv.org/pdf/1911.12835.pdf .
        shape2 : float
            delta in https://arxiv.org/pdf/1911.12835.pdf .

        Returns
        -------
        f : array_like
            Probability density values.
        """

        x = np.array(x)
        # In Matlab syntax: f = delta .* beta ./ alpha .* (x ./ alpha).^
        # (beta - 1) .* (1 - exp(-1 * (x ./ alpha).^beta)).^(delta - 1) .*
        # exp(-1 .* (x ./ alpha).^beta);
        a = scale
        b = shape
        d = shape2
        term1 = d * np.divide(b, a) * np.power(np.divide(x, a), b - 1)
        term2 = np.power(1 - np.exp(-1 * np.power(np.divide(x, a), b)), d - 1)
        term3 = np.exp(-1 * np.power(np.divide(x, a), b))
        f = np.multiply(term1, np.multiply(term2, term3))
        f = np.array(f) # Ensure that f is an numpy array, also if x is 1D.

        # Ensure that PDF(negative value) = 0
        f[x < 0] = 0

        return f

    def fit(self, sample, method='WLS', shape=None, loc=None, scale=None, shape2=None):
        """

        Parameters
        ----------
        sample : array_like,
            The data that should be used to fit the distribution to it.
        method : str,
            Fitting method, either 'MLE' for maximum likelihood estimation or
            'WLS' for weighted least squares, similar to: https://arxiv.org/pdf/1911.12835.pdf
        shape : float, optional
            If given the shape parameter won't be fitted.
        loc : Not used, must be None
            The exponentiated Weibull distribution does not have a location
            parameter. However, for consistency we have it as a parameter at the
            expected place.
        scale : float, optional
            If given the scale parameter won't be fitted.
        shape2 : float, optional
            If given the second shape parametr won't be fitted.

        Returns
        -------
        params: 4-dimensional tuple
         Holds (shape, loc=None, scale, shape2).

        """
        def estimateAlphaBetaWithWLS(delta, xi, pi, do_return_parameters=True):
            """
            Translated from the Matlab implementation available at
            https://github.com/ahaselsteiner/exponentiated-weibull/blob/issue%231/ExponentiatedWeibull.m#L210

            Parameters
            ----------
            delta : float
                shape2 parameter of the distribution
            xi : array_like
                Sorted sample.
            pi : array_like
                Probabilities of the sorted sample.

            Returns
            -------
            (WLSError, pHat) where pHat are the parameter estimates.
            """
            xi = np.array(xi)
            pi = np.array(pi)


            # As xi = 0 causes problems when xstar is calculated, zero-elements
            # are not considered in the parameter estimation.
            indices = np.nonzero(xi)
            xi = xi[indices]
            pi = pi[indices]


            # First, transform xi and pi to get a lienar relationship.
            xstar_i = np.log10(xi)
            power_term = np.power(pi, np.divide(1.0, delta))
            pstar_i = np.log10(-1.0 * np.log(1.0 - power_term))

            # Define the weights.
            wi = np.divide(np.power(xi, 2), sum(np.power(xi, 2)))

            # Estimate the parameters alphaHat and betaHat.
            pstarbar = sum(np.multiply(wi, pstar_i))
            xstarbar = sum(np.multiply(wi, xstar_i))
            temp1 = sum(np.multiply(wi, np.multiply(pstar_i, xstar_i))) - np.multiply(pstarbar, xstarbar)
            temp2 = sum(np.multiply(wi, np.power(pstar_i, 2))) - np.power(pstarbar, 2)
            bHat = np.divide(temp1, temp2)
            aHat = xstarbar - bHat * pstarbar
            alphaHat = np.power(10, aHat)
            betaHat = 1.0 / bHat
            pHat = (alphaHat, betaHat, delta)

            # Compute the weighted least squares error.
            xiHat = np.multiply(alphaHat, np.power(np.multiply(-1, np.log(1 - np.power(pi, 1.0 / delta))), 1.0 / betaHat))
            WLSError = sum(np.multiply(wi, np.power(xi - xiHat, 2.0)))

            if do_return_parameters:
                return pHat, WLSError
            else:
                return WLSError # If the function shall be used as cost function.

        params = (shape, loc, scale, shape2)
        # Code written based on the Matlab implementation available here:
        # https://github.com/ahaselsteiner/exponentiated-weibull/blob/issue%231/ExponentiatedWeibull.m
        isFixed = (shape is not None, loc is not None, scale is not None, shape2 is not None)
        if method == 'WLS': # Weighted least squares
            n = sample.size
            i = np.array(range(n)) + 1
            pi = np.divide((i - 0.5), n)
            xi = np.sort(sample)
            delta0 = 2
            if sum(isFixed) == 0:
                shape2 = scipy.optimize.fmin(estimateAlphaBetaWithWLS, delta0, args=(xi, pi, False))
                shape2 = shape2[0] # Returns an 1x1 array, however, we want a float.
                pHat, WLSError = estimateAlphaBetaWithWLS(shape2, xi, pi)
            elif sum(isFixed) == 1:
                if isFixed[3] == 1:
                       pHat, WLSError = estimateAlphaBetaWithWLS(shape2, xi, pi)
                else:
                    err_msg = "Error. Fixing shape or scale is not implemented yet."
                    raise NotImplementedError(err_msg)
            elif sum(isFixed == 2):
                err_msg = "Error. Fixing multiple parameters is not implemented yet."
                raise NotImplementedError(err_msg)
            else:
                err_msg = "Error. At least one parameter needs to be free to fit it."
                raise NotImplementedError(err_msg)

        params = (pHat[1], loc, pHat[0], shape2) # shape, location, scale, shape2
        constantParams = (ConstantParam(pHat[1]), # shape parameter.
                          ConstantParam(loc), # location parameter.
                          ConstantParam(pHat[0]), # scale parameter.
                          ConstantParam(shape2)) # Second shape parameter

        self.__init__(*constantParams)

        return params


class LognormalDistribution(ParametricDistribution):
    """
    A Lognormal distribution.

    There are two different ways to create the Lognormal distribution. You can either use the parameters ``sigma`` and
    ``mu`` as *kwargs* or the parameters ``shape, None, scale`` as *args*.

    Examples
    --------
    Create a LognormalDistribution and plot the cumulative distribution function,
    using explicit ``sigma`` and ``mu`` arguments:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from viroconcom.params import ConstantParam
    >>> my_sigma = ConstantParam(1)
    >>> my_mu = ConstantParam(0)
    >>> dist = LognormalDistribution(sigma=my_sigma, mu=my_mu)
    >>> x = np.linspace(0, 10, num=100)
    >>> #example_plot = plt.plot(x, dist.cdf(x, None, (None, None, None)),\
                                            #label='Lognormal(mu, sigma)')

    Creating the same LognormalDistribution using the ``shape`` and ``scale`` parameters:

    >>> shape = ConstantParam(1)
    >>> scale = ConstantParam(1)  # scale = exp(mu) = exp(0) = 1
    >>> dist = LognormalDistribution(shape, None, scale)
    >>> x = np.linspace(0, 10, num=100)
    >>> #example_plot = plt.plot(x, dist.cdf(x, None, (None, None, None)),\
                                            #label="Lognormal (shape, scale)")

    """

    def __init__(self, shape=None, loc=None, scale=None, **kwargs):
        saved_args = locals()

        loc = None
        if "sigma" in kwargs and "mu" in kwargs:
            self.sigma = kwargs["sigma"]
            self.mu = kwargs["mu"]

            shape = self.sigma
            # Make mu a scale parameter
            if isinstance(self.mu, FunctionParam):
                _func = self.mu._func
                _a = self.mu.a
                _b = self.mu.b
                _c = self.mu.c
                # Keep possibly already existing wrapper
                scale_wrapper = Wrapper(np.exp, self.mu._wrapper)
                # Create new FunctionParam so the passed one does not get altered
                scale = FunctionParam(self.mu.func_name, _a, _b, _c, wrapper=scale_wrapper)
                scale._func = _func
            else:
                scale = ConstantParam(np.exp(self.mu(None)))

        super().__init__(shape, loc, scale)
        self.name = "Lognormal"

        self._valid_shape = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }
        self._valid_scale = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }



    def _scipy_cdf(self, x, shape, _, scale):
        return sts.lognorm.cdf(x, s=shape, scale=scale)

    def _scipy_i_cdf(self, probabilities, shape, _, scale):
        return sts.lognorm.ppf(probabilities, s=shape, scale=scale)

    def __str__(self):
        if hasattr(self, "mu"):
            return  "LognormalDistribution with shape={}, loc={}," \
                    "scale={}, mu={}.".format(self.shape, self.loc,
                                             self.scale, self.mu)
        else:
            return  "LognormalDistribution with shape={}, loc={}," \
                    "scale={}.".format(self.shape, self.loc, self.scale)


class NormalDistribution(ParametricDistribution):
    """
    A Normal distribution.

    The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.

    Examples
    --------
    Create a NormalDistribution and plot the cumulative distribution function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from viroconcom.params import ConstantParam
    >>> loc = ConstantParam(0)
    >>> scale = ConstantParam(1)
    >>> dist = NormalDistribution(None, loc, scale)
    >>> x = np.linspace(0, 5, num=100)
    >>> #example_plot = plt.plot(x, dist.cdf(x, None, (None, None, None)),\
                                #label="Normal")

    """

    def __init__(self, shape=None, loc=None, scale=None):
        super().__init__(shape, loc, scale)
        self.name = "Normal"
        self._valid_scale = {"min" : 0, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }

    def _scipy_cdf(self, x, _, loc, scale):
        return sts.norm.cdf(x, loc=loc, scale=scale)

    def _scipy_i_cdf(self, probabilities, _, loc, scale):
        return sts.norm.ppf(probabilities, loc=loc, scale=scale)


class MultivariateDistribution():
    """
    A Multivariate distribution consisting of multiple univariate distributions and dependencies.

    Attributes
    ----------
    distributions : list of Distribution
        A list containing the distributions.
    dependencies : list of tuples
        A list containing a dependency tuple for each distribution.
    n_dim : int
        The number of distributions. Equal to len(distributions).
    """

    def __init__(self, distributions=None, dependencies=None):
        """
        Parameters
        ----------
        distributions : list of Distribution
            A list containing distributions.
        dependencies : list of tuples
            A list with one dependency tuple for each distribution.
        """
        self.distributions = []
        self.dependencies = []
        self.n_dim = 0

        if not distributions is None:
            self.add_distributions(distributions, dependencies)


    def add_distributions(self, distributions, dependencies):
        """
        Add one or multiple distributions and define dependencies.

        Parameters
        ----------
        distributions : ``Distribution`` or list of ``Distribution``
            A distribution or list containing distributions.
        dependencies : tuple or list of tuples
            A dependency tuple or list with one dependency tuple for each distribution.

        """
        backup = (self.distributions, self.dependencies, self.n_dim)

        try:
            it = iter(distributions)
            dist_is_iter = True
        except TypeError:
            dist_is_iter = False
        try:
            # Check if dependencies is a list of tuples.
            it = iter(dependencies[0])
            dep_is_iter_of_tuple = True
        except TypeError:
            dep_is_iter_of_tuple = False

        if dist_is_iter != dep_is_iter_of_tuple:
            raise ValueError("If distributions is iterable, so has to be dependencies "
                             "and vise versa.")
        if dist_is_iter:
            if len(distributions) != len(dependencies):
                raise ValueError(("distributions and dependencies must be of the same length, "
                                  "but where len(distributions)={} and len(dependencies)={}."
                                  "".format(len(distributions), len(dependencies))))

            for i in range(len(distributions)):
                self.distributions.append(distributions[i])
                self.dependencies.append(dependencies[i])


        else:
            self.distributions.append(distributions)
            self.dependencies.append(dependencies)

        self.n_dim += len(self.distributions)

        err_msg = self._check_dependencies(dep_is_iter_of_tuple)
        if err_msg is not None:
            self.distributions, self.dependencies, self.n_dim = backup
            raise ValueError(err_msg)

    def _check_dependencies(self, dep_is_iter_of_tuple):
        """
        Make sure the dependencies are valid.

        e.g. a RV can only depend on RV's that appear in order before itself.
        """
        for dimension, dependency in enumerate(self.dependencies):
            if(dep_is_iter_of_tuple):
                if len(dependency) < 3:
                    return ("The length of the dependency in dimension '{}' was less than three.".format(dimension))
                elif not all([True if d is None or d < dimension else False for d in dependency]):
                    return ("The dependency of dimension '{}' must have smaller index than dimension or 'None'.".format(dimension))
                elif not all([True if d is None or d >= 0 else False for d in dependency]):
                    return ("The dependency of dimension '{}' must be positive or 'None'.".format(dimension))
            elif len(self.dependencies) < 3:
                return ("The length of dependencies was less than three.")
        return None

    def cell_averaged_joint_pdf(self, coords):
        """
        Calculates the cell averaged joint probabilty density function.

        Multiplies the cell averaged probability densities of all distributions.

        Parameters
        ----------
        coords : array_like
            List of the sampling points of the random variables.
            The length of coords has to equal self.n_dim.

        Returns
        -------
        fbar : ndarray
            Cell averaged joint probabilty density function evaluated at coords.
            It is a self.n_dim dimensional array,
            with shape (len(coords[0]), len(coords[1]), ...)

        """
        fbar = np.ones(((1,) * self.n_dim), dtype=np.float64)
        for dist_index in range(self.n_dim):
            fbar = np.multiply(fbar, self.cell_averaged_pdf(dist_index, coords))

        return fbar

    def cell_averaged_pdf(self, dist_index, coords):
        """
        Calculates the cell averaged probabilty density function of a single distribution.

        Calculates the pdf by approximating it with the finite differential quotient
        of the cumulative distributions function, evaluated at the grid cells borders.
        i.e. :math:`f(x) \\approx \\frac{F(x+ 0.5\\Delta x) - F(x- 0.5\\Delta x) }{\\Delta x}`

        Parameters
        ----------
        dist_index : int
            The index of the distribution to calculate the pdf of,
            according to order of self.distributions.
        coords : array_like
            List of the sampling points of the random variables.
            The pdf is calculated at coords[dist_index].
            The length of coords has to equal self.n_dim.

        Returns
        -------
        fbar : ndarray
            Cell averaged probabilty density function evaluated at coords[dist_index].
            It is a self.n_dim dimensional array.
        """
        assert(len(coords) == self.n_dim)
        dimensions = range(self.n_dim)
        dist = self.distributions[dist_index]
        dependency = self.dependencies[dist_index]
        cdf = dist.cdf

        dx = coords[dist_index][1] - coords[dist_index][0]

        fbar_shape = tuple((len(coords[i]) for i in dimensions if i <= dist_index))
        fbar = np.zeros(fbar_shape)

        # Iterate over all possible dependencies.
        iter_ranges = (range(i) for i in fbar_shape[0:-1])
        it = itertools.product(*iter_ranges)

        for multi_index in it:
            f_index = multi_index + (slice(None),)  # = e.g. (i,j,:) for 3 dimensions

            current_point = np.empty(len(coords))
            for i in range(len(coords)):
                if i < len(multi_index):
                    current_point[i] = coords[i][multi_index[i]]
                else:  # random variable must be independent of this dimensions, so set to 0
                    current_point[i] = 0

            # calculate averaged pdf
            lower = cdf(coords[dist_index] - 0.5 * dx, current_point, dependency)
            upper = cdf(coords[dist_index] + 0.5 * dx, current_point, dependency)
            fbar[f_index] = (upper - lower)  # / dx

        # Append axes until self.n_dim is reached.
        n_dim_shape = fbar_shape + tuple((1 for i in range(self.n_dim - len(fbar_shape))))
        fbar = fbar.reshape(n_dim_shape)
        return fbar / dx

    def latex_repr(self, var_symbols=None):
        """
        Returns the joint probabilty density function in latex format.

        Parameters
        ----------
        var_symbols : list
            List of the random variable symbols, the first letter should be
             capitalized and further characters will be converged to subscripts,
             an example would be  ['Hs', 'Tp', 'V']

        Returns
        -------
        latex_string : str
            The joint pdf in latex format (without $)
            E.g. f(h_s,t_p)=f_{H_s}(h_s)=
        """
        # Constants to name the scale, shape and location parameter
        wbl_scale = r"\alpha"
        wbl_shape = r"\beta"
        wbl_loc = r"\gamma"

        if not var_symbols:
            var_symbols=[]
            for i in range(self.n_dim):
                var_symbols.append("X_{" + str(i) + "}")
        else:
            for i in range(self.n_dim):
                var_symbols[i] = var_symbols[i][0] + "_{" + \
                                 var_symbols[i][1:] + "}"

        # Realization symbols are not capitalized, e.g. hs for the
        # realization of Hs
        # Next line, thanks to: https://stackoverflow.com/questions/3840843/
        # how-to-downcase-the-first-character-of-a-string
        downcase_first_char = lambda s: s[:1].lower() + s[1:] if s else ''
        realization_symbols = []
        for i in range(self.n_dim):
            realization_symbols.append(downcase_first_char(var_symbols[i]))

        joint_pdf_all_symbols_w_commas = ""
        for i in range(self.n_dim):
            joint_pdf_all_symbols_w_commas += realization_symbols[i] + ","
        joint_pdf_all_symbols_w_commas = joint_pdf_all_symbols_w_commas[:-1]

        latex_string_list = []
        latex_string = r"\text{ joint PDF: }"
        latex_string_list.append(latex_string)
        latex_string = "f(" + joint_pdf_all_symbols_w_commas + ")="
        left_side_pdfs = ["" for x in range(self.n_dim)]
        for i in range(self.n_dim):
            left_side_pdfs[i] += "f_{" + var_symbols[i]
            # If there is at least one depedent parameter.
            if not all(x is None for x in self.dependencies[i]):
                left_side_pdfs[i] += "|"
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        left_side_pdfs[i] += var_symbols[j] + ','
                left_side_pdfs[i] = left_side_pdfs[i][:-1]
            left_side_pdfs[i] += "}(" + realization_symbols[i]
            # If there is at least one depedent parameter.
            if not all(x is None for x in self.dependencies[i]):
                left_side_pdfs[i] += "|"
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        left_side_pdfs[i] += realization_symbols[j] + ','
                left_side_pdfs[i] = left_side_pdfs[i][:-1]
            left_side_pdfs[i] += ")"
            latex_string += left_side_pdfs[i]
        latex_string_list.append(latex_string)

        for i in range(self.n_dim):
            latex_string = ""
            latex_string_list.append(latex_string) # add a blank line
            latex_string_list.append(str(i+1) + r"\text{. variable, }" +
                                     str(var_symbols[i]) + ": ")
            latex_string = left_side_pdfs[i] + "="
            scale_name = None
            shape_name = None
            loc_name = None
            if self.distributions[i].name == "Weibull":
                scale_name = wbl_scale + r"_{" + realization_symbols[i] + "}"
                shape_name = wbl_shape + r"_{" + realization_symbols[i] + "}"
                loc_name = wbl_loc + r"_{" + realization_symbols[i] + "}"
                latex_string += r"\dfrac{" + wbl_shape + r"_{" + \
                                realization_symbols[i] + \
                                r"}}{" + wbl_scale + r"_{" + \
                                realization_symbols[i] \
                        + "}}\left(\dfrac{" + realization_symbols[i] + \
                                r"-" +  loc_name + r"}{" + wbl_scale + r"_{" + \
                                realization_symbols[i] \
                        + r"}}\right)^{" + wbl_shape + r"_{" + \
                                realization_symbols[i] + \
                                r"}-1}\exp\left[-\left(\dfrac{" + \
                                realization_symbols[i] \
                        + r"-" + loc_name + r"}{" + wbl_scale + r"_{" + \
                                realization_symbols[i] + \
                                r"}}\right)^{" + wbl_shape + r"_{" + \
                                realization_symbols[i] +\
                                r"}}\right]"
            elif self.distributions[i].name == "Normal":
                scale_name = r"\sigma_{" + realization_symbols[i] + "}"
                loc_name = r"\mu_{" + realization_symbols[i] + "}"
                latex_string += r"\dfrac{1}{\sqrt{2\pi" + scale_name + r"^2}}"\
                                + r"\exp\left[-\dfrac{(" + \
                                realization_symbols[i] + r"-" + loc_name + \
                                r")^2}{2" + scale_name + r"^2}\right]"
            elif self.distributions[i].name == "Lognormal":
                shape_name = r"\tilde{\sigma}_{" + realization_symbols[i] + "}"

                # Here for simplicity we save mu in the 'scale_value'
                # variable, although we know scale = exp(mu)
                if hasattr(self.distributions[i], 'mu'):
                    scale_name = r"\tilde{\mu}_{" + realization_symbols[i] + "}"
                else:
                    scale_name = r"\exp{\tilde{\mu}}_{" + realization_symbols[i] + "}"

                latex_string += r"\dfrac{1}{" + realization_symbols[i] + \
                                r"\tilde{\sigma}_{" \
                        + realization_symbols[i] + \
                                r"}\sqrt{2\pi}}\exp\left[-\dfrac{(\ln " + \
                                realization_symbols[i] \
                                + r"-\tilde{\mu}_{" + realization_symbols[i] + \
                                r"})^2}{2\tilde{\sigma}_{" \
                        + realization_symbols[i] + r"}^2}\right]"
            latex_string_list.append(latex_string)
            if scale_name:
                latex_string = r"\quad\text{ with }"
                if self.distributions[i].name == "Lognormal":

                    if hasattr(self.distributions[i], 'mu'):
                        # Here for simplicity we save mu in the 'scale_value'
                        # variable, although we know scale = exp(mu)
                        scale_value = str(self.distributions[i].mu)
                    else:
                        scale_value = str(self.distributions[i].scale)

                else:
                    scale_value = str(self.distributions[i].scale)
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        scale_value = scale_value.replace(
                            'x', realization_symbols[j])
                latex_string += scale_name + "=" + scale_value + ","
                latex_string_list.append(latex_string)
            if shape_name:
                if scale_name:
                    latex_string = r"\quad\qquad\;\; "
                else:
                    latex_string = r"\quad\text{ with }"

                shape_value = str(self.distributions[i].shape)
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        shape_value = shape_value.replace(
                            'x', realization_symbols[j])
                latex_string += shape_name + "=" + shape_value
                if loc_name:
                    latex_string += ","
                else:
                    latex_string += "."
                latex_string_list.append(latex_string)
            if loc_name:
                latex_string = r"\quad\qquad\;\; "
                loc_value = str(self.distributions[i].loc)
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        loc_value = loc_value.replace(
                            'x', realization_symbols[j])
                latex_string += loc_name + "=" + loc_value + "."
                latex_string_list.append(latex_string)
        return latex_string_list


class KernelDensityDistribution(Distribution):
    """
    A kernel density distribution.

    Examples
    --------
    Create a KernelDensityDistribution:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # ------------ part from fitting.py --------------
    >>> import statsmodels.api as sm
    >>> sample = np.random.RandomState(500)
    >>> p = sample.normal(10, 1, 500)
    >>> dens = sm.nonparametric.KDEUnivariate(p)
    >>> dens.fit(gridsize=2000)
    >>> list = (dens.cdf, dens.icdf)
    >>> # ------------------------------------------------
    >>> dist = KernelDensityDistribution(list)
    >>> x = np.linspace(0, 5, num=100)
    >>> #example_plot = plt.plot(x, dist.cdf(x, None, (None, None, None)),\
                                #label='KernelDensity')


    """

    def __init__(self, params):
        """
        Represents a Kernel Density distribution by using two lists that contain coordinates which
        represent the cdf and icdf distribution. The Kernel Densitiy Distribution is created by the fitting process and
        can then be used to build a contour.

        Note
        ----
        There are no parameters such as shape, loc, scale used for the Kernel Density Distribution.
        Therefor it can not be dependent.

        Parameters
        ----------
        params : list,
            Contains cdf coordinates on index 0 and icdf coordinates on index 1
            params[0] -> cdf
            params[1] -> icdf
        """

        self.name = "KernelDensity"
        self._cdf = params[0]
        self._i_cdf = params[1]

    def cdf(self, x, rv_values, dependencies):
        """
        Calculate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to calculate the cdf.
        rv_values : array_like
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If x is an array, M must be len(x).
            --Not used for Kernel Density--
        dependencies : tuple
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.
            --Not used for Kernel Density--

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x.
        """
        result = []
        for point in x:
            # scale x
            x_point = point * (len(self._cdf) - 1) / (max(self._i_cdf) - min(self._i_cdf))
            # use linear fit if x_point is between two points
            linear_fit = np.poly1d(np.polyfit([int(x_point), int(x_point) + 1],
                                              [self._cdf[int(x_point)],
                                               self._cdf[int(x_point) + 1]], 1))
            result.append(linear_fit(x_point))
        return result

    def i_cdf(self, probability, rv_values, dependencies):
        """
        Calculate percent-point function. (inverse cumulative distribution function)

        Parameters
        ----------
        probabilities : array_like
            Probabilities for which to calculate the i_cdf.
        rv_values : array_like
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If probabilities is an array, M must be len(probabilities).
            --Not used for Kernel Density--
        dependencies : tuple
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.
            --Not used for Kernel Density--

        Returns
        -------
        i_cdf : ndarray,
            Inverse cumulative distribution function evaluated for probabilities.
        """
        result = []
        for point in probability:
            # scale probability
            x_point = point * (len(self._i_cdf) - 1)
            # use linear fit if x_point is between two points
            linear_fit = np.poly1d(np.polyfit([int(x_point), int(x_point) + 1],
                                              [self._i_cdf[int(x_point)],
                                               self._i_cdf[int(x_point) + 1]], 1))
            result.append(linear_fit(x_point))
        return result


if __name__ == "__main__":
    import doctest
    doctest.testmod()
