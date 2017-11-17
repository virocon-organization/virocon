#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uni- and multivariate distributions.
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as sts

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
    shape : Param,
        The shape parameter.
    loc : Param,
        The location parameter.
    scale : Param,
        The scale parameter.
    name : str,
        The name of the distribution. ("Weibull", "LogNormal", Normal)
    _scipy_cdf : function,
        The cumulative distribution function from scipy. (sts.weibull_min.cdf, ...)
    _scipy_i_cdf : function,
        The inverse cumulative distribution (or percent-point) function.(sts.weibull_min.ppf, ...)
    _default_shape : float
        The default shape parameter.
    _default_loc : float
        The default loc parameter.
    _default_scale : float
        The default scale parameter.


    Notes
    -----
    The following attributes/methods need to be initialised by child classes:
        - name
        - _scipy_cdf
        - _scipy_i_cdf
    """

    @abstractmethod
    def __init__(self, shape, loc, scale):
        """
        Parameters
        ----------
        shape : Param,
            The shape parameter.
        loc : Param,
            The location parameter.
        scale : Param,
            The scale parameter.
        """
        self.shape = shape
        self.loc = loc
        self.scale = scale
        # the following attributes need to be overwritten by subclasses
        self.name = "Parametric"  # e.g. "Weibull", "Lognormal",  ...

        self._default_shape = 1
        self._default_loc = 0
        self._default_scale = 1

        self._valid_shape = {"min" : -np.inf, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True } # -inf < shape < inf
        self._valid_loc = {"min" : -np.inf, "strict_greater" : True,
                           "max" : np.inf, "strict_less" : True }
        self._valid_scale = {"min" : -np.inf, "strict_greater" : True,
                             "max" : np.inf, "strict_less" : True }


    @abstractmethod
    def _scipy_cdf(self, x, shape, loc, scale):
        """Overwrite with appropriate cdf function from scipy package. """

    @abstractmethod
    def _scipy_i_cdf(self, probabilities, shape, loc, scale):
        """Overwrite with appropriate i_cdf function from scipy package. """

    def cdf(self, x, rv_values, dependencies):
        """
        Calculate the cumulative distribution function.

        Parameters
        ----------
        x : array_like,
            Points at which to calculate the cdf.
        rv_values : array_like,
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If x is an array, M must be len(x).
        dependencies : tuple,
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.


        Returns
        -------
        cdf : ndarray,
            Cumulative distribution function evaluated at x under condition rv_values.
        """

        shape_val, loc_val, scale_val = self._get_parameter_values(rv_values, dependencies)

        return self._scipy_cdf(x, shape_val, loc_val, scale_val)

    def i_cdf(self, probabilities, rv_values, dependencies):
        """
        Calculate percent-point function. (inverse cumulative distribution function)

        Parameters
        ----------
        probabilities : array_like,
            Probabilities for which to calculate the i_cdf.
        rv_values : array_like,
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If probabilities is an array, M must be len(probabilities).
        dependencies : tuple,
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.

        Returns
        -------
        i_cdf : ndarray,
            Inverse cumulative distribution function evaluated for probabilities
            under condition rv_values.
        """

        shape_val, loc_val, scale_val = self._get_parameter_values(rv_values, dependencies)

        return self._scipy_i_cdf(probabilities, shape_val, loc_val, scale_val)

    def _get_parameter_values(self, rv_values, dependencies):
        """
        Evaluates the conditional shape, loc, scale parameters.

        Parameters
        ----------
        rv_values : array_like,
            Values of all random variables in variable space in correct order.
        dependencies : tuple,
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.

        Returns
        -------
        parameter_vals : tuple,
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The tuple contains the values of the parameters evaluated under the conditions
            of dependencies.
            The values are either float or lists of float.
        """
        #get values
        params = (self.shape, self.loc, self.scale)
        defaults = (self._default_shape, self._default_loc, self._default_scale)
        parameter_vals = []

        for i, param in enumerate(params):
            if param is None:
                parameter_vals.append(defaults[i])
            elif dependencies[i] is None:
                parameter_vals.append(param(None))
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
        param_index : int,
            Index of parameter.
            (0 = 'shape', 1 = 'loc', 2 = 'scale')
        param_value : float,
            Value of parameter.

        Raises
        ------
        ValueError
            If parameter value is outside the boundaries.
        """

        if param_index == 0:
            valid = self._valid_shape
            param_name = "shape"
        elif param_index == 1:
            valid = self._valid_loc
            param_name = "loc"
        elif param_index == 2:
            valid = self._valid_shape
            param_name = "scale"

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



class WeibullDistribution(ParametricDistribution):
    """
    A Weibull distribution.

    Examples
    --------
    Create a WeibullDistribution and plot the cumulative distribution function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from compute.params import ConstantParam
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
    >>> from compute.params import ConstantParam
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

        loc = None
        if "sigma" in kwargs and "mu" in kwargs:
            self.sigma = kwargs["sigma"]
            self.mu = kwargs["mu"]

            shape = self.sigma
            # make mu a scale parameter
            if isinstance(self.mu, FunctionParam):
                _func = self.mu._func
                _a = self.mu.a
                _b = self.mu.b
                _c = self.mu.c
                # keep possibly already existing wrapper
                scale_wrapper = Wrapper(np.exp, self.mu._wrapper)
                # create new FunctionParam so the passed one does not get altered
                scale = FunctionParam(_a, _b, _c, "f1", wrapper=scale_wrapper)
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


class NormalDistribution(ParametricDistribution):
    """
    A Normal distribution.

    Examples
    --------
    Create a NormalDistribution and plot the cumulative distribution function:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from compute.params import ConstantParam
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
    distributions : list of Distribution,
        A list containing the distributions.
    dependencies : list of tuples,
        A list containing a dependency tuple for each distribution.
    n_dim : int,
        The number of distributions. Equal to len(distributions).
    """

    def __init__(self, distributions=None, dependencies=None):
        """
        Parameters
        ----------
        distributions : list of Distribution,
            A list containing distributions.
        dependencies : list of tuples,
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
        distributions : ``Distribution`` or list of ``Distribution``,
            A distribution or list containing distributions.
        dependencies : tuple or list of tuples,
            A dependency tuple or list with one dependency tuple for each distribution.

        """
        backup = (self.distributions, self.dependencies, self.n_dim)

        try:
            it = iter(distributions)
            dist_is_iter = True
        except TypeError:
            dist_is_iter = False
        try:
            #check if indexable ~ list of tuple
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
                if len(dependency) != 3:
                    return ("The length of the dependency in dimension '{}' was not three.".format(dimension))
                elif not all([True if d is None or d < dimension else False for d in dependency]):
                    return ("The dependency of dimension '{}' must have smaller index than dimension or 'None'.".format(dimension))
                elif not all([True if d is None or d >= 0 else False for d in dependency]):
                    return ("The dependency of dimension '{}' must be positive or 'None'.".format(dimension))
            elif len(self.dependencies) != 3:
                return ("The length of dependencies was not three.")
        return None

    def cell_averaged_joint_pdf(self, coords):
        """
        Calculates the cell averaged joint probabilty density function.

        Multiplies the cell averaged probability densities of all distributions.

        Parameters
        ----------
        coords : array_like,
            List of the sampling points of the random variables.
            The length of coords has to equal self.n_dim.

        Returns
        -------
        fbar : ndarray,
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
        dist_index : int,
            The index of the distribution to calculate the pdf of,
            according to order of self.distributions.
        coords : array_like,
            List of the sampling points of the random variables.
            The pdf is calculated at coords[dist_index].
            The length of coords has to equal self.n_dim.

        Returns
        -------
        fbar : ndarray,
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

        # iterate over all possible dependencies
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

        # append axes until self.n_dim is reached
        n_dim_shape = fbar_shape + tuple((1 for i in range(self.n_dim - len(fbar_shape))))
        fbar = fbar.reshape(n_dim_shape)
        return fbar / dx

    def getPdfAsLatexString(self, var_symbols=None):
        """
        Returns the joint probabilty density function in latex format.

        Parameters
        ----------
        var_symbols : list,
            List of the random variable symbols, the first letter should be
             capitalized and further characters will be converged to subscripts,
             an example would be  ['Hs', 'Tp', 'V']

        Returns
        -------
        latex_string : String,
            The joint pdf in latex format (without $)
            E.g. f(h_s,t_p)=f_{H_s}(h_s)=
        """
        if not var_symbols:
            var_symbols=[]
            for i in range(self.n_dim):
                var_symbols.append("X_{" + str(i) + "}")
        else:
            for i in range(self.n_dim):
                var_symbols[i] = var_symbols[i][0] + "_{" + var_symbols[i][1:] + "}"

        # realization symbols are not capitalized, e.g. hs for the realization of Hs
        downcase_first_char = lambda s: s[:1].lower() + s[1:] if s else '' # thanks to: https://stackoverflow.com/questions/3840843/how-to-downcase-the-first-character-of-a-string
        realization_symbols = []
        for i in range(self.n_dim):
            realization_symbols.append(downcase_first_char(var_symbols[i]))

        joint_pdf_all_symbols_w_commas = ""
        for i in range(self.n_dim):
            joint_pdf_all_symbols_w_commas += realization_symbols[i] + ","
        joint_pdf_all_symbols_w_commas = joint_pdf_all_symbols_w_commas[:-1]

        latex_string = "f(" + joint_pdf_all_symbols_w_commas + ")="
        left_side_pdfs = ["" for x in range(self.n_dim)]
        for i in range(self.n_dim):
            left_side_pdfs[i] += "f_{" + var_symbols[i]
            if not all(x is None for x in self.dependencies[i]): # if there is at least one depedent paramter
                left_side_pdfs[i] += "|"
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        left_side_pdfs[i] += var_symbols[j] + ','
                left_side_pdfs[i] = left_side_pdfs[i][:-1]
            left_side_pdfs[i] += "}(" + realization_symbols[i]
            if not all(x is None for x in self.dependencies[i]): # if there is at least one depedent paramter
                left_side_pdfs[i] += "|"
                for j in range(self.n_dim):
                    if  j in self.dependencies[i]:
                        left_side_pdfs[i] += realization_symbols[j] + ','
                left_side_pdfs[i] = left_side_pdfs[i][:-1]
            left_side_pdfs[i] += ")"
            latex_string += left_side_pdfs[i]
        #latex_string += r"\\"
        latex_string += r"\text{ with }"
        latex_string += r"\dfrac{a}{b}"
        print(latex_string)
        return latex_string

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
        x : array_like,
            Points at which to calculate the cdf.
        rv_values : array_like,
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If x is an array, M must be len(x).
            --Not used for Kernel Density--
        dependencies : tuple,
            A 3-element tuple with one entry each for the shape, loc and scale parameters.
            The entry is the index of the random variable the parameter depends on.
            The index order has to be the same as in rv_values.
            --Not used for Kernel Density--

        Returns
        -------
        cdf : ndarray,
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
        probabilities : array_like,
            Probabilities for which to calculate the i_cdf.
        rv_values : array_like,
            Values of all random variables in variable space in correct order.
            This can be a 1-dimensional array with length equal to the number of
            random variables N or a 2-dimensional array with shape (N, M).
            If probabilities is an array, M must be len(probabilities).
            --Not used for Kernel Density--
        dependencies : tuple,
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