#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits distributions to data.
"""

import warnings
import time
import numpy as np

from multiprocessing import Pool, TimeoutError
from numbers import Number
import statsmodels.api as sm
import scipy.stats as sts
from scipy.optimize import curve_fit, minimize
from inspect import signature

from .settings import (SHAPE_STRING, LOCATION_STRING, SCALE_STRING,
                       SHAPE2_STRING,
                       LOGNORMAL_EXPMU_PARAMETER_KEYWORD,
                       LOGNORMAL_MU_PARAMETER_KEYWORD,
                       NORMAL_KEYWORD, WEIBULL_3P_KEYWORD,
                       WEIBULL_3P_KEYWORD_ALTERNATIVE,
                       WEIBULL_2P_KEYWORD, WEIBULL_EXP_KEYWORD,
                       INVERSE_GAUSSIAN_KEYWORD)
from .params import ConstantParam, FunctionParam
from .distributions import (WeibullDistribution, ExponentiatedWeibullDistribution,
                            LognormalDistribution, NormalDistribution,
                            InverseGaussianDistribution,
                            KernelDensityDistribution, MultivariateDistribution)


__all__ = ["Fit"]


# Dependence functions for the parameters, the following functions are available:
# A 3-parameter power function (a dependence function).
def _power3(x, a, b, c):
    return a + b * x ** c


# A 3-parameter exponential function (a dependence function).
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)


# Logarithmic square function. Function has two paramters, but 3 are given such
# that in the software all dependence functions can be called with 3 parameters.
def _lnsquare2(x, a, b, c):
    return np.log(a + b * np.sqrt(np.divide(x, 9.81)))


# 3-parameter function that decreases with x to the power of c (a dependence fucntion).
def _powerdecrease3(x, a, b, c):
    return a + 1 / (x + b) ** c


# 3-parameter function that asymptotically decreases (a dependence function).
def _asymdecrease3(x, a, b, c):
    return a + b / (1 + c * x)


# A 4-parameter logististics function (a dependence function).
def _logistics4(x, a, b, c, d):
    return a + b / (1 + np.exp(-1 * np.abs(c) * (x - d)))


# A 3-parameter function designed for the scale parameter (alpha) of an
# exponentiated Weibull distribution with shape2=5 (see 'Global hierarchical
# models for wind and wave contours').
def _alpha3(x, a, b, c, C1=None, C2=None, C3=None, C4=None):
    return (a + b * x ** c) \
           / 2.0445 ** (1 / _logistics4(x, C1, C2, C3, C4))
           
# A 3-parameter 2nd order polynomial  (a dependence function).
def _poly2(x, a, b, c):
    return a * x**2 + b * x + c

# A 2-parameter 1st order polynomial  (a dependence function).
def _poly1(x, a, b):
    return a * x + b


# Bounds for function parameters:
# 0 < a < inf
# 0 < b < inf
# -inf < c < inf
# 0 < d < inf
_bounds = ([np.finfo(np.float64).tiny, np.finfo(np.float64).tiny, -np.inf, np.finfo(np.float64).tiny],
          [np.inf, np.inf, np.inf, np.inf])


class BasicFit():
    """
    Holds the parameters (shape, loc, scale, shape2) and the raw data of a single fit.

    Attributes
    ----------
    shape : float
        The shape parameter for the fit.

    loc : float
        The location parameter for the fit.

    scale : float
        The scale parameter for the fit.

    shape2 : float, defaults to None
        The second shape parameter for the fit.

    samples : list of float, defaults to None
        The raw data that was used for this fit. For that case that there is no dependency this
        list contains the whole data of the dimension.

    """

    def __init__(self, shape, loc, scale, shape2=None, samples=None):

        # parameters for the distribution
        if type(shape) == ConstantParam:
            self.shape = shape(0)
        elif isinstance(shape, Number):
            self.shape = shape
        else:
            err_msg = "Parameter 'shape' must be an instance of Number or type of ConstantParam " \
                      "but was '{}'.".format(type(shape))
            raise TypeError(err_msg)
        if type(loc) == ConstantParam:
            self.loc = loc(0)
        elif isinstance(loc, Number):
            self.loc = loc
        else:
            err_msg = "Parameter 'loc' must be an instance of Number or type of ConstantParam " \
                      "but was '{}'.".format(type(loc))
            raise TypeError(err_msg)
        if type(scale) == ConstantParam:
            self.scale = scale(0)
        elif isinstance(scale, Number):
            self.scale = scale
        else:
            err_msg = "Parameter 'scale' must be an instance of Number or type of ConstantParam " \
                      "but was '{}'.".format(type(scale))
            raise TypeError(err_msg)
        if type(shape2) == ConstantParam:
            self.shape2 = shape2(0)
        elif isinstance(shape, Number):
            self.shape2 = shape2
        elif shape2 is None:
            self.shape2 = None
        else:
            err_msg = "Parameter 'shape2' must be an instance of Number or type of ConstantParam " \
                      "but was '{}'.".format(type(shape))
            raise TypeError(err_msg)


        # Raw data
        self.samples = samples

    def __str__(self):
        return "BasicFit with shape={}, loc={}, scale={}, shape2={}.".format(
            self.shape, self.loc, self.scale, self.shape2)


class FitInspectionData():
    """
    This class holds information for plotting the fits of a single dimension. It is used to give
    a visual look about how good the fits in this dimension were.

    Attributes
    ----------
    used_number_of_intervals : int
        The actually number of intervals this dimension is divided for other dependent dimensions.

    shape_at : list of float
        Values of the interval centers of the parent variable that were used to
        fit the shape parameter.

    shape_value : list of float
        The associated values of the parameter shape to the divided dimension the shape
        parameter depends on.

    loc_at : list of float
        Values of the interval centers of the parent variable that were used to
        fit the location parameter.

    loc_value : list of float
        The associated values of the parameter loc to the divided dimension the location
        parameter depends on.

    scale_at : list of float
        Values of the interval centers of the parent variable that were used to
        fit the scale parameter.

    scale_value : list of float
        The associated values of the parameter scale to the divided dimension the scale
        parameter depends on.

    shape2_at : list of float
        Values of the interval centers of the parent variable that were used to
        fit the shape2 parameter.

    shape2_value : list of float
        Values of the parameter shape2 at the nth interval.

    shape_samples : list of list
        This list with the length of the number of used intervals for the shape parameter
        contains lists with the used samples for the respective fit.

    loc_samples : list of list
        This list with the length of the number of used intervals for the location parameter
        contains lists with the used samples for the respective fit.

    scale_samples : list of list
        This list with the length of the number of used intervals for the scale parameter
        contains lists with the used samples for the respective fit.

    """

    def __init__(self):

        # Number of the intervals this dimension is divided
        self.used_number_of_intervals = None

        # Parameter values and the data they belong to
        self.shape_at = None
        self._shape_value = [[], [], [], []]

        self.loc_at = None
        self._loc_value = [[], [], [], []]

        self.scale_at = None
        self._scale_value = [[], [], [], []]

        self.shape2_at = None
        self._shape2_value = [[], [], [], []]

        # Raw data for each parameter of this dimension
        self.shape_samples = []
        self.loc_samples = []
        self.scale_samples = []
        self.shape2_samples = []

    @property
    def shape_value(self):
        """
        Takes out the list that contains the shape parameters.

        Returns
        -------
        list of float
             The associated values of the parameter shape to the divided dimension the shape
             parameter depends on.
        Notes
        ------
        This function can be used as attribute.
        """
        return self._shape_value[0]

    @property
    def loc_value(self):
        """
        Takes out the list that contains the location parameters.

        Returns
        -------
        list of float
             The associated values of the parameter loc to the divided dimension the location
             parameter depends on.
        Notes
        ------
        This function can be used as attribute.
        """
        return self._loc_value[1]

    @property
    def scale_value(self):
        """
        Takes out the list that contains the scale parameters.

        Returns
        -------
        list of float
             The associated values of the parameter scale to the divided dimension the scale
             parameter depends on.
        Notes
        ------
        This function can be used as attribute.
        """
        return self._scale_value[2]

    @property
    def shape2_value(self):
        """
        Takes out the list that contains the shape2 parameters.

        Returns
        -------
        list of float
             The associated values of the parameter shape2 to the divided
             dimension the shape2 parameter depends on.
        Notes
        ------
        This function can be used as attribute.
        """
        return self._shape2_value[3]

    def get_dependent_param_points(self, param):
        """
        This function can be used to get the param_at and the param_value lists as tuple for a
        given parameter.

        Parameters
        ----------
        param : str
            The respective parameter.
        Returns
        -------
        tuple of list
             The param_at and the param_value.
        Raises
        ------
        ValueError
            If the parameter is unknown.
        """
        if param == SHAPE_STRING:
            return self.shape_at, self.shape_value
        elif param == LOCATION_STRING:
            return self.loc_at, self.loc_value
        elif param == SCALE_STRING:
            return self.scale_at, self.scale_value
        elif param == SHAPE2_STRING:
            return self.shape2_at, self.shape2_value
        else:
            err_msg = "Parameter '{}' is unknown.".format(param)
            raise ValueError(err_msg)

    def append_basic_fit(self, param ,basic_fit):
        """
        This function can be used to add a single fit to the hold data.

        Parameters
        ----------
        param : str
            The respective parameter the data should be associated.
        basic_fit : BasicFit
            The data of the single fit hold in a BasicData object.

        Raises
        ------
        ValueError
            If the parameter is unknown.
        """
        if param == SHAPE_STRING:
            self._shape_value[0].append(basic_fit.shape)
            self._shape_value[1].append(basic_fit.loc)
            self._shape_value[2].append(basic_fit.scale)
            self._shape_value[3].append(basic_fit.shape2)
            self.shape_samples.append(basic_fit.samples)
        elif param == LOCATION_STRING:
            self._loc_value[0].append(basic_fit.shape)
            self._loc_value[1].append(basic_fit.loc)
            self._loc_value[2].append(basic_fit.scale)
            self._loc_value[3].append(basic_fit.shape2)
            self.loc_samples.append(basic_fit.samples)
        elif param == SCALE_STRING:
            self._scale_value[0].append(basic_fit.shape)
            self._scale_value[1].append(basic_fit.loc)
            self._scale_value[2].append(basic_fit.scale)
            self._scale_value[3].append(basic_fit.shape2)
            self.scale_samples.append(basic_fit.samples)
        elif param == SHAPE2_STRING:
            self._shape2_value[0].append(basic_fit.shape)
            self._shape2_value[1].append(basic_fit.loc)
            self._shape2_value[2].append(basic_fit.scale)
            self._shape2_value[3].append(basic_fit.shape2)
            self.shape_samples.append(basic_fit.samples)
        else:
            err_msg = "Parameter '{}' is unknown.".format(param)
            raise ValueError(err_msg)

    def get_basic_fit(self, param, index):
        """
        This function returns the data of a single fit to a given parameter and the index of the
        interval of the divided dimension the parameter depends on.

        Parameters
        ----------
        param : str
            The respective parameter of the data.
        index : int
            The index of the interval.
        Returns
        -------
        BasicFit
             The data of the single fit hold in a BasicData object.
        Raises
        ------
        ValueError
            If the parameter is unknown.
        """
        if param == SHAPE_STRING:
            return BasicFit(shape=self._shape_value[0][index],
                            loc=self._shape_value[1][index],
                            scale=self._shape_value[2][index],
                            shape2=self._shape_value[3][index],
                            samples=self.shape_samples[index])
        elif param == LOCATION_STRING:
            return BasicFit(shape=self._loc_value[0][index],
                            loc=self._loc_value[1][index],
                            scale=self._loc_value[2][index],
                            shape2=self._shape_value[3][index],
                            samples=self.loc_samples[index])
        elif param == SCALE_STRING:
            return BasicFit(shape=self._scale_value[0][index],
                            loc=self._scale_value[1][index],
                            scale=self._scale_value[2][index],
                            shape2=self._shape_value[3][index],
                            samples=self.scale_samples[index])
        else:
            err_msg = "Parameter '{}' is unknown.".format(param)
            raise ValueError(err_msg)


class Fit():
    """
    Holds data and information about a fit.

    Note
    ----
    The fitted results are not checked for correctness. The created distributions may not contain
    useful parameters. Distribution parameters are being checked in the contour creation process.

    Attributes
    ----------
    mul_var_dist : MultivariateDistribution
        Distribution that is calculated

    multiple_fit_inspection_data : list of FitInspectionData
        Contains fit inspection data objects for each dimension.

    Examples
    --------
    Create a Fit and visualize the result in a IFORM contour:

    >>> from multiprocessing import Pool
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> import scipy.stats as sts
    >>> from scipy.optimize import curve_fit
    >>> from viroconcom.params import ConstantParam, FunctionParam
    >>> from viroconcom.distributions import (WeibullDistribution,\
                                           LognormalDistribution,\
                                           NormalDistribution,\
                                           KernelDensityDistribution,\
                                           MultivariateDistribution)
    >>> from viroconcom.contours import IFormContour
    >>> prng = np.random.RandomState(42)
    >>> sample_1 = prng.normal(10, 1, 500)
    >>> sample_2 = [point + prng.uniform(-5, 5) for point in sample_1]
    >>> dist_description_1 = {'name': 'KernelDensity', 'dependency': (None, None, None), 'number_of_intervals': 5}
    >>> dist_description_2 = {'name': 'Normal', 'dependency': (None, 0, None), 'functions':(None, 'power3', None)}
    >>> my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2))
    >>> my_contour = IFormContour(my_fit.mul_var_dist)
    >>> #example_plot = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IForm")


    Create a Fit and visualize the result in a HDC contour:

    >>> from viroconcom.contours import HighestDensityContour
    >>> sample_3 = prng.weibull(2, 500) + 15
    >>> sample_4 = [point + prng.uniform(-1, 1) for point in sample_1]
    >>> dist_description_1 = {'name': 'Weibull', 'dependency': (None, None, None)}
    >>> dist_description_2 = {'name': 'Normal', 'dependency': (None, None, None)}
    >>> my_fit = Fit((sample_3, sample_4), (dist_description_1, dist_description_2))
    >>> return_period = 50
    >>> state_duration = 3
    >>> limits = [(0, 20), (0, 20)]
    >>> deltas = [0.05, 0.05]
    >>> my_contour = HighestDensityContour(my_fit.mul_var_dist, return_period, state_duration, limits, deltas,)
    >>> #example_plot2 = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="HDC")


    An Example how to visualize how good your fit is:

    >>> dist_description_0 = {'name': 'Weibull_3p', 'dependency': (None, None, None), 'number_of_intervals': 3}
    >>> dist_description_1 = {'name': 'Lognormal', 'dependency': (None, None, 0), 'functions': (None, None, 'exp3')}
    >>> my_fit = Fit((sample_1, sample_2), (dist_description_0, dist_description_1))
    >>>
    >>> #fig = plt.figure(figsize=(10, 8))
    >>> #example_text = fig.suptitle("Dependence of 'scale'")
    >>>
    >>> #ax1 = fig.add_subplot(221)
    >>> #title1 = ax1.set_title("Fitted curve")
    >>> param_grid = my_fit.multiple_fit_inspection_data[1].scale_at
    >>> x_1 = np.linspace(5, 15, 100)
    >>> #ax1_plot = ax1.plot(param_grid, my_fit.multiple_fit_inspection_data[0].scale_value, 'x')
    >>> #example_plot1 = ax1.plot(x_1, my_fit.mul_var_dist.distributions[1].scale(x_1))
    >>>
    >>> #ax2 = fig.add_subplot(222)
    >>> #title2 = ax2.set_title("Distribution '1'")
    >>> #ax2_hist = ax2.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[0], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[0])
    >>> #ax2_plot = ax2.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
    >>>
    >>> #ax3 = fig.add_subplot(223)
    >>> #title3 = ax3.set_title("Distribution '2'")
    >>> #ax3_hist = ax3.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[1], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[1])
    >>> #ax3_plot = ax3.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
    >>>
    >>> #ax4 = fig.add_subplot(224)
    >>> #title4 = ax4.set_title("Distribution '3'")
    >>> #ax4_hist = ax4.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[2], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[2])
    >>> #ax4_plot = ax4.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))

    """

    def __init__(self, samples, dist_descriptions, timeout=None):
        """
        Creates a Fit, by estimating the parameters of the distribution.

        Parameters
        ----------
        samples : tuple or list of list
            List that contains data to be fitted : samples[0] -> first variable (i.e. wave height)
                                                   samples[1] -> second variable
                                                   ...
        dist_descriptions : list of dict
            contains dictionary for each parameter. See note for further information.

        timeout : int, optional
            The maximum time in seconds there the contour has to be computed.
            This parameter also controls multiprocessing. If timeout is None
            serial processing is performed, if it is not None multiprocessing
            is used. Defaults to None.

        Raises
        ------
        TimeoutError
            If the calculation takes too long and the given value for timeout is exceeded.

        Note
        ----
        dist_descriptions contains the following keys where some are
        required and some are optional.

        Required:

        name : str
            name of distribution (defined in settings.py):

            - Weibull_2p,
            - Weibull_3p,
            - Weibull_Exp
            - Lognormal (shape, scale),
            - Lognormal_SigmaMu (sigma, mu),
            - Normal,
            - InverseGaussian
            - KernelDensity (no dependency)

        dependency : tuple or list of int
            Length of 3 or 4 in the order (shape, loc, scale, shape2) contains:

            - None -> no dependency
            - int -> depends on particular dimension

        functions : tuple or list of str
            Length of 3 in the order : (shape, loc, scale), usable options:

            - :power3: :math:`a + b * x^c`
            - :exp3: :math:`a + b * e^{x * c}`
            - :lnsquare2: :math:`ln[a + b * sqrt(x / 9.81)`
            - :powerdecrease3: :math:`a + 1 / (x + b)^c`
            - :asymdecrease3: :math:`a + b / (1 + c * x)`
            - :logistics4: :math:`a + b / [1 + e^{-1 * |c| * (x - d)}]`
            - remark : in case of Lognormal_SigmaMu it is (sigma, None, mu)

        Optional:

        fixed_parameters : tuple of floats
            If some parameters shall not be estimated, but should be fixed,
            they can be specified with this key. Floats are interpeted in the
            order (shape, location, scale, shape2).

        do_use_weights_for_dependence_function : Boolean, defaults to False
            If true the dependence function is fitted using weights. The weights
            are 1 / parameter_value such that a normalization is performed.

        and either number_of_intervals, width_of_intervals or points_per_interval:

        number_of_intervals : int
            Number of bins the data of this variable should be seperated for fits which depend
            upon it. If the number of bins is given, the width of the bins is determined
            automatically.

        width_of_intervals : float
            Width of the bins. When the width of the bins is given, the number of bins is
            determined automatically.
            
        points_per_interval : int
            The number of points per interval.

        """

        # If the distribution is 1D and the user did not create a list or tuple
        # of length 1, let's create it
        if type(dist_descriptions) not in [list,tuple] and \
                        type(dist_descriptions.get('name')) is str:
            if len(dist_descriptions) != len(samples):
                samples = (samples, )
            dist_descriptions = (dist_descriptions, )


        self.dist_descriptions = dist_descriptions # Compute references this attribute at plot.py

        list_number_of_intervals = []
        list_width_of_intervals = []
        list_points_per_interval = []
        for dist_description in dist_descriptions:
            if (dist_description.get('number_of_intervals') is None 
                and dist_description.get('width_of_intervals') is None
                and dist_description.get('points_per_interval') is None):
                dist_description['number_of_intervals'] = 15
            list_number_of_intervals.append(dist_description.get('number_of_intervals'))
            list_width_of_intervals.append(dist_description.get('width_of_intervals'))
            list_points_per_interval.append(dist_description.get('points_per_interval'))
        for dist_description in dist_descriptions:
            dist_description['list_number_of_intervals'] = list_number_of_intervals
            dist_description['list_width_of_intervals'] = list_width_of_intervals
            dist_description['list_points_per_interval'] = list_points_per_interval

        # Results will be computed for each dimension
        multiple_results = []
        self.multiple_fit_inspection_data = []
        distributions = []
        dependencies = []

        for dimension in range(len(samples)):
            dist_description = dist_descriptions[dimension]

            # Use multiprocessing if a timeout is defined.
            if timeout:
                pool = Pool()
                multiple_results.append(
                    pool.apply_async(self._get_distribution,
                                     (dimension, samples),
                                     dist_description)
                )

            else:
                kwargs = dist_description
                distribution, dependency, used_number_of_intervals, \
                fit_inspection_data = self._get_distribution(
                    dimension=dimension,
                    samples=samples,
                    **kwargs)
                distributions.append(distribution)
                dependencies.append(dependency)

                # Save the used number of intervals
                for dep_index, dep in enumerate(dependency):
                    if dep is not None:
                        self.dist_descriptions[dep][
                            'used_number_of_intervals'] = \
                            used_number_of_intervals[dep_index]

                self.multiple_fit_inspection_data.append(fit_inspection_data)

        # If multiprocessing is used we have to collect the results differently.
        if timeout:
            # Define start time
            start_time = time.time()
            # Get distributions
            for i, res in enumerate(multiple_results):
                current_time = time.time()
                time_difference = current_time - start_time # Previous used time
                try:
                    distribution, dependency, used_number_of_intervals, fit_inspection_data = res.get(
                        timeout=timeout-time_difference)
                except TimeoutError:
                    err_msg = "The calculation takes too long. " \
                              "It takes longer than the given " \
                              "value for a timeout, " \
                              "which is '{} seconds'.".format(timeout)
                    raise TimeoutError(err_msg)

                # Saves distribution and dependency for particular dimension
                distributions.append(distribution)
                dependencies.append(dependency)

                # Add fit inspection data for current dimension
                self.multiple_fit_inspection_data.append(fit_inspection_data)

                # Save the used number of intervals
                for dep_index, dep in enumerate(dependency):
                    if dep is not None:
                        self.dist_descriptions[dep]['used_number_of_intervals'] = \
                            used_number_of_intervals[dep_index]

            # Add used number of intervals for dimensions with no dependency
            for fit_inspection_data in self.multiple_fit_inspection_data:
                if not fit_inspection_data.used_number_of_intervals:
                    fit_inspection_data.used_number_of_intervals = 1

        # Save multivariate distribution
        self.mul_var_dist = MultivariateDistribution(distributions, dependencies)

    @staticmethod
    def _fit_distribution(sample, name, fixed_parameters=(None, None, None, None)):
        """
        Fits the distribution and returns the parameters.

        Parameters
        ----------
        sample : list of float
            Raw data the distribution is fitted on.
        name : str
            Name of the distribution ("Weibull_2p", "Weibull_3p", "Lognormal" or
            "Lognormal_SigmaMu", "Normal", "KernelDensity"). They keyword list
            is defined in settings.py.
        fixed_parameters : tuple of float
            Specifies which value parameters are fixed and thus are not
            fitted. None means that it is not fixed, but shall be estimated.
        Returns
        -------
        tuple of ConstantParam
             The computed parameters in the order of (shape, loc, scale, shape2).
        Raises
        ------
        ValueError
            If the distribution is unknown.
        """
        if fixed_parameters != (None, None, None, None) and \
                        name != WEIBULL_EXP_KEYWORD:
            err_msg = "Fixing parameters is not implemented for the " \
                      "distribution {} yet.".format(name)
            raise NotImplementedError(err_msg)
        if name == WEIBULL_2P_KEYWORD:
            # Do not fit the location parameter because it is 0 for a 2-p. dist.
            params = sts.weibull_min.fit(sample, floc=0)
        elif name == WEIBULL_3P_KEYWORD or \
                        name == WEIBULL_3P_KEYWORD_ALTERNATIVE:
            params = sts.weibull_min.fit(sample)
            if params[1] < 0:
                warnings.warn('The estimated location parameter of a translated '
                              'Weibull distribution was negative ({}). However, '
                              'as this is likely unphysical and could lead to '
                              'problems with conditonal variables, the '
                              'location parameter is set to 0.'.format(params[1]),
                              RuntimeWarning, stacklevel=2)
                params = (params[0], 0, params[2])
        elif name == WEIBULL_EXP_KEYWORD:
            dist = ExponentiatedWeibullDistribution()
            params = dist.fit(sample, shape=fixed_parameters[0],
                                  scale=fixed_parameters[1],
                                  loc=fixed_parameters[2],
                                  shape2=fixed_parameters[3])
        elif name == NORMAL_KEYWORD:
            params = list(sts.norm.fit(sample))
            # Shape doesn't exist for normal
            params.insert(0, 0)
        elif name == LOGNORMAL_EXPMU_PARAMETER_KEYWORD or \
                        name == LOGNORMAL_MU_PARAMETER_KEYWORD:
            # For the lognormal distribution the value of the location parameter is set to 0.
            params = sts.lognorm.fit(sample, floc=0)
        elif name == INVERSE_GAUSSIAN_KEYWORD:
            # For the inverse Gaussian distribution the value of the location parameter is set to 0.
            params = sts.invgauss.fit(sample, floc=0)
        elif name == 'KernelDensity':
            dens = sm.nonparametric.KDEUnivariate(sample)
            dens.fit(gridsize=2000)
            # Kernel density doesn't have shape, loc, scale
            return (dens.cdf, dens.icdf)
        else:
            err_msg = "Distribution '{}' is unknown.".format(name)
            raise ValueError(err_msg)

        if len(params) == 3:
            constant_params = (ConstantParam(params[0]),
                    ConstantParam(params[1]),
                    ConstantParam(params[2]),
                    ConstantParam(None))
        elif len(params) == 4:
            constant_params = (ConstantParam(params[0]),
                    ConstantParam(params[1]),
                    ConstantParam(params[2]),
                    ConstantParam(params[3]))
        else:
            err_msg = "params must have a length of 4, but was '{}'."\
                .format(len(params))
            raise ValueError(err_msg)

        return constant_params

    @staticmethod
    def _get_function(function_name):
        """
        Returns the function.

        Parameters
        ----------
        function_name : str
            Options are 'power3', 'exp3', 'lnsquare2', 'powerdecrease3',
            'asymdecrease3', 'logistics4', 'alpha3'.

        Returns
        -------
        func
             The actual function named function_name.

        Raises
        ------
        ValueError
            If the function is unknown.
        """

        if function_name == 'power3':
            return _power3
        elif function_name == 'exp3':
            return _exp3
        elif function_name == 'lnsquare2':
            return _lnsquare2
        elif function_name == 'powerdecrease3':
            return _powerdecrease3
        elif function_name == 'asymdecrease3':
            return _asymdecrease3
        elif function_name == 'logistics4':
            return _logistics4
        elif function_name == 'alpha3':
            return _alpha3
        elif function_name == 'poly2':
            return _poly2
        elif function_name == 'poly1':
            return _poly1
        elif function_name is None:
            return None
        else:
            err_msg = "Function '{}' is unknown.".format(function_name)
            raise ValueError(err_msg)

    @staticmethod
    def _append_params(name, param_values, dependency, index, sample, fixed_parameters=(None, None, None, None)):
        """
        Distributions are being fitted and the results are appended to param_points.

        Parameters
        ----------
        name : str
            Name of distribution (e.g. 'Weibull_2p' or 'Lognormal').
        param_values : list of list,
            Contains lists that contain values for each param : order (shape, loc, scale).
        dependency : list of int
            Length of 3 in the order (shape, loc, scale) contains :
            None -> no dependency
            int -> depends on particular dimension
        index : int
            The current parameter as int in the order of (shape, loc, scale) (i.e. 0 -> shape).
        sample : list of float
            Values that are used to fit the distribution.
        fixed_parameters : tuple of float
            Specifies which value parameters are fixed and thus are not
            fitted. None means that it is not fixed, but shall be estimated.

        Returns
        -------
        BasicFit
             The information of this single fit.
        """

        # Fit distribution
        current_params = Fit._fit_distribution(sample, name, fixed_parameters=fixed_parameters)

        # Create basic fit object
        basic_fit = BasicFit(*current_params, sample)

        for i in range(index, len(dependency)):
            # Check if there is a dependency and whether it is the right one
            if dependency[i] is not None and \
                            dependency[i] == dependency[index]:
                # Calculated parameter is appended to param_values
                param_values[i].append(current_params[i])
        return basic_fit

    @staticmethod
    def _get_fitting_values(sample, samples, name, dependency, index,
                            number_of_intervals=None, bin_width=None,
                            points_per_interval=None,
                            min_datapoints_for_fit=20,
                            fixed_parameters=(None, None, None, None)):
        """
        Returns values for fitting.

        Parameters
        ----------
        sample : list of float
            The current sample to fit.
        samples : list of list
            List that contains data to be fitted : samples[0] -> first variable (i.e. wave height)
                                                   samples[1] -> second variable
                                                   ...
        name : str
            Name of distribution (e.g. 'Weibull_2p' or 'Lognormal').
        dependency : list of int
            Length of 3 in the order (shape, loc, scale) contains :
                None -> no dependency
                int -> depends on particular dimension
        index : int
            Order : (shape, loc, scale) (i.e. 0 -> shape).
        number_of_intervals : int
            Number of distributions used to fit shape, loc, scale.
        min_datapoints_for_fit : int
            Minimum number of datapoints required to perform the fit.
        fixed_parameters : tuple of float
            Specifies which value parameters are fixed and thus are not
            fitted. None means that it is not fixed, but shall be estimated.
        Notes
        -----
        For that case that number_of_intervals and also bin_width is given the parameter
        number_of_intervals is used.

        Returns
        -------
        interval_centers : ndarray
            Array with length of the number of bins that contains the centers of the
            calculated bins.
        dist_values : list of list
            List with length of the number of intervals that contains for each bin center
            the used samples for the current fit.
        param_values : list of list
            List with length of three that contains for each parameter (shape, loc, scale)
            a list with length of the number of bins that contains the calculated parameters.
        multiple_basic_fit : list of BasicFit
            Contains information for each fit.
        Raises
        ------
        RuntimeError
            If the parameter number_of_intervals or bin_width or points_per_interval was not specified.
        RuntimeError
            If there was not enough data and the number of intervals was less than three.
        """

        # Sort samples.
        stacked_samples = np.stack((sample, samples[dependency[index]])).T
        sort_indice = np.argsort(stacked_samples[:, 1])
        sorted_samples = stacked_samples[sort_indice]

        # Compute intervals.
        if number_of_intervals is not None:
            interval_centers, interval_width = np.linspace(
                min(samples[dependency[index]]), max(samples[dependency[index]]),
                num=number_of_intervals, endpoint=False, retstep=True)
            interval_centers += 0.5 * interval_width
        elif bin_width is not None:
            interval_width = bin_width
            interval_centers = np.arange(
                0.5 * interval_width,
                max(samples[dependency[index]]) + 0.5 * interval_width,
                interval_width)
        elif points_per_interval is not None:
            n_full_chunks = np.floor(len(sorted_samples) / points_per_interval)
            last_full_chunk_idx = int(n_full_chunks * points_per_interval)
            full_sample_chunks = np.split(sorted_samples[:last_full_chunk_idx],
                                          n_full_chunks)
            remaining_chunk = sorted_samples[last_full_chunk_idx:]
            # Use the mean as it is used by ESSC
            # (https://github.com/WEC-Sim/WDRT/blob/master/WDRT/ESSC.py),
            # I would prefer median though
            full_chunk_centers = np.mean(np.array(full_sample_chunks)[:, :, 1], axis=-1)
            remaining_chunk_center = np.mean(remaining_chunk[:, 1])
            
            sample_chunks = full_sample_chunks
            sample_chunks.append(remaining_chunk)
            interval_centers = np.append(full_chunk_centers, [remaining_chunk_center])
            
        else:
            raise RuntimeError(
                "Either the parameters number_of_intervals or bin_width or "
                "points_per_interval has to be specified, "
                "otherwise the intervals are not specified. Exiting.")


        

        # Return values.
        param_values = [[], [], []]
        dist_values = []

        # List of all basic fits.
        multiple_basic_fit = []

        # Deleted interval_centers by index.
        deleted_centers = []

        # Define the data interval that is used for the fit.
        for i, step in enumerate(interval_centers):
            if points_per_interval is not None:
                # samples_in_interval = sample_chunks[i]
                samples_in_interval = np.sort(sample_chunks[i][:, 0])
            else:
                mask = ((sorted_samples[:, 1] >= step - 0.5 * interval_width) &
                        (sorted_samples[:, 1] < step + 0.5 * interval_width))
                samples_in_interval = sorted_samples[mask, 0]
            if len(samples_in_interval) >= min_datapoints_for_fit:
                try:
                    # Fit distribution to selected data.
                    basic_fit = Fit._append_params(name,
                                                   param_values,
                                                   dependency,
                                                   index,
                                                   samples_in_interval,
                                                   fixed_parameters=fixed_parameters)
                    multiple_basic_fit.append(basic_fit)
                    dist_values.append(samples_in_interval)
                except ValueError:
                    deleted_centers.append(i) # Add index of unused center.
                    warnings.warn(
                        "A ValueError occured for the interval centered at '{}'"
                        " in dimension '{}'."
                            .format(step, dependency[index]),
                        RuntimeWarning, stacklevel=2)
            else:
                # For case that too few fitting data for the step were found
                # the step is deleted.
                deleted_centers.append(i) # Add index of unused center.


        # Delete interval centers that were not used.
        interval_centers = np.delete(interval_centers, deleted_centers)
        if len(interval_centers) < 3:
            nr_of_intervals = str(len(interval_centers))
            raise RuntimeError("Your settings resulted in " + nr_of_intervals +
                               " intervals. However, at least 3 intervals are "
                               "required. Consider changing the required  "
                               " minimum of datapoints within an interval using "
                               "the 'min_datapoints_for_fit' key.")

        return interval_centers, dist_values, param_values, multiple_basic_fit

    def _get_distribution(self, dimension, samples, **kwargs):
        """
        Returns the fitted distribution, the dependency and information to
        visualize all fits for this dimension.

        Parameters
        ----------
        dimension : int
            Number of the variable. For example it can be 0, which means that
            this is the first variable (for example sig. wave height).
        samples : list of list
            List that contains data to be fitted :
            samples[0] -> first variable (for example sig. wave height)
            samples[1] -> second variable (for example spectral peak period)
            ...
        **kwargs: contains the fit_description data to clarify which kind of
            distribution with which method should be fitted.
        Returns
        -------
        distribution : Distribution
            The fitted distribution instance.
        dependency : list of int
            List that contains the used dependencies for fitting.
        used_number_of_intervals: list of int
            List with length of three that contains the used number of intervals
            for each parameter (shape, loc, scale).
        fit_inspection_data : FitInspectionData
            Object that holds information about all fits in this dimension.
        Raises
        ------
        NotImplementedError
            If the the name of a dependent distribution was 'KernelDensity'.
        RuntimeError
            If not a good fit was found.
        """

        # Save settings for distribution
        sample = samples[dimension]
        if 'name' in kwargs:
            name = kwargs.get('name')
        else:
            err_msg = "_get_distribution misses the argument 'name'."
            raise TypeError(err_msg)
        dependency = kwargs.get('dependency', (None, None, None, None))
        functions = kwargs.get('functions', ('polynomial', )*len(dependency))
        list_number_of_intervals = kwargs.get('list_number_of_intervals')
        list_width_of_intervals = kwargs.get('list_width_of_intervals')
        list_points_per_interval =  kwargs.get('list_points_per_interval')
        min_datapoints_for_fit = kwargs.get('min_datapoints_for_fit', 20)
        fixed_parameters = kwargs.get('fixed_parameters', (None, None, None, None))
        do_use_weights_for_dependence_function = kwargs.get('do_use_weights_for_dependence_function', False)

        # Fit inspection data for current dimension
        fit_inspection_data = FitInspectionData()

        # Initialize used_number_of_intervals (shape, loc, scale, shape2)
        used_number_of_intervals = [None, None, None, None]

        # Handle KernelDensity separated
        if name == 'KernelDensity':
            if not all(x is None for x in dependency):
                raise NotImplementedError("KernelDensity can not be conditional.")
            return KernelDensityDistribution(Fit._fit_distribution(sample, name)), dependency, \
                   used_number_of_intervals, fit_inspection_data

        # Initialize params (shape, loc, scale, shape2). The second shape
        # parameter is currently only used by the exponentiated Weibull distr.
        params = [None, None, None, None]

        for index in range(len(dependency)):

            # Continue if params is yet computed
            if params[index] is not None:
                continue

            # In case that there is no dependency for this param
            if dependency[index] is None:
                current_params = Fit._fit_distribution(sample, name, fixed_parameters=fixed_parameters)

                # Basic fit for no dependency
                basic_fit = BasicFit(*current_params, sample)
                for i in range(index, len(functions)):
                    # Check if the other parameters have also no dependency
                    if dependency[i] is None:

                        # Add basic fit to fit inspection data
                        if i == 0:
                            fit_inspection_data.append_basic_fit(SHAPE_STRING,
                                                                 basic_fit)
                        elif i == 1:
                            fit_inspection_data.append_basic_fit(LOCATION_STRING,
                                                                 basic_fit)
                        elif i == 2:
                            fit_inspection_data.append_basic_fit(SCALE_STRING,
                                                                 basic_fit)
                        elif i == 3:
                            fit_inspection_data.append_basic_fit(SHAPE2_STRING,
                                                                 basic_fit)

                        if i == 2 and name == LOGNORMAL_MU_PARAMETER_KEYWORD:
                            params[i] = ConstantParam(np.log(current_params[i](0)))
                        else:
                            params[i] = current_params[i]
            # In case that there is a dependency
            else:
                # If the number of intervals is given.
                if list_number_of_intervals[dependency[index]]:
                    interval_centers, dist_values, param_values, multiple_basic_fit = \
                        Fit._get_fitting_values(
                            sample, samples, name, dependency, index,
                            number_of_intervals=list_number_of_intervals[dependency[index]],
                            min_datapoints_for_fit=min_datapoints_for_fit,
                            fixed_parameters=fixed_parameters)
                # If a the (constant) width of the intervals is given.
                elif list_width_of_intervals[dependency[index]]:
                    interval_centers, dist_values, param_values, multiple_basic_fit = \
                        Fit._get_fitting_values(
                            sample, samples, name, dependency, index,
                            bin_width=list_width_of_intervals[dependency[index]],
                            min_datapoints_for_fit=min_datapoints_for_fit,
                            fixed_parameters=fixed_parameters)
                elif list_points_per_interval[dependency[index]]:
                    interval_centers, dist_values, param_values, multiple_basic_fit = \
                        Fit._get_fitting_values(
                            sample, samples, name, dependency, index,
                            points_per_interval=list_points_per_interval[dependency[index]],
                            min_datapoints_for_fit=min_datapoints_for_fit,
                            fixed_parameters=fixed_parameters)

                for i in range(index, len(functions)):
                    # Check if the other parameters have the same dependency
                    if dependency[i] is not None and dependency[i] == dependency[index]:
                        # Add basic fits to fit inspection data
                        for basic_fit in multiple_basic_fit:
                            if i == 0:
                                fit_inspection_data.append_basic_fit(
                                    SHAPE_STRING,
                                    basic_fit)
                            elif i == 1:
                                fit_inspection_data.append_basic_fit(
                                    LOCATION_STRING,
                                    basic_fit)
                            elif i == 2:
                                fit_inspection_data.append_basic_fit(
                                    SCALE_STRING,
                                    basic_fit)
                            if i == 3:
                                fit_inspection_data.append_basic_fit(
                                    SHAPE2_STRING,
                                    basic_fit)

                        # Add interval centers to fit inspection data
                        if i == 0:
                            fit_inspection_data.shape_at = interval_centers
                        elif i == 1:
                            fit_inspection_data.loc_at = interval_centers
                        elif i == 2:
                            fit_inspection_data.scale_at = interval_centers
                        elif i == 3:
                            fit_inspection_data.shape2_at = interval_centers

                        # Add used number of intervals for current parameter
                        used_number_of_intervals[i] = len(interval_centers)

                        if i == 2 and name == LOGNORMAL_MU_PARAMETER_KEYWORD:
                            fit_points = [np.log(p(None)) for p in param_values[i]]
                        else:
                            fit_points = [p(None) for p in param_values[i]]
                        # Fit parameters with particular dependence function.
                        try:
                            # Get the number of parameters of the dependence function
                            # and choose the according bounds for the fit.
                            sig = signature(Fit._get_function(functions[i]))
                            nParam = 0
                            for param in sig.parameters.values():
                                if param.kind == param.POSITIONAL_OR_KEYWORD and \
                                                param.default is param.empty:
                                    nParam = nParam + 1
                            bLower = _bounds[0][0: nParam - 1]
                            bUpper = _bounds[1][0: nParam - 1]
                            bounds = (bLower, bUpper)


                            if functions[i] == "alpha3":
                                # alpha3 is handled differently, since it
                                # depends on a prevously fitted logistics4
                                # function.

                                # Get the fitted coefficients for the shape parameter,
                                # which is modelled with a logistics4 function.
                                f = params[0]
                                C1 = f.a
                                C2 = f.b
                                C3 = f.c
                                C4 = f.d

                                # The lambda function was used based on https://stackoverflow.com/
                                # questions/47884910/fixing-fit-parameters-in-curve-fit
                                if f.func_name == "logistics4":
                                    if do_use_weights_for_dependence_function:
                                        param_popt, param_pcov = \
                                            curve_fit(
                                                lambda x, a, b,
                                                c: Fit._get_function(functions[i])(x, a, b, c,
                                                                                   C1=C1, C2=C2, C3=C3, C4=C4),
                                                interval_centers, fit_points,
                                                sigma=fit_points, bounds=bounds)
                                    else:
                                        param_popt, param_pcov = \
                                            curve_fit(
                                                lambda x, a, b,
                                                c: Fit._get_function(functions[i])(x, a, b, c,
                                                                                   C1=C1, C2=C2, C3=C3, C4=C4),
                                                interval_centers, fit_points, bounds=bounds)
                                else:
                                    err_msg = \
                                        "The alpha3 function is only " \
                                        "allowed when shape is modelled " \
                                        "with a logistics4 function. In your " \
                                        "model shape is modelled with a function " \
                                        "of type '{}'.".format(f.func_name)
                                    raise TypeError(err_msg)

                            elif functions[i] == "poly2":
                                # This is based on the fitting described in
                                # Eckert-Gallup2016: https://doi.org/10.1016/j.oceaneng.2015.12.018.

                                if do_use_weights_for_dependence_function:
                                    raise NotImplementedError("do_use_weights_for_dependence_function "
                                                              "is not implemented for poly2")

                                x = interval_centers
                                y = fit_points
                                def error_func(p):
                                    return np.sum((_poly2(x, p[0], p[1], p[2]) - y)**2) 
                                
                                ineq_cons = {"type": "ineq",
                                             "fun": lambda x: np.array([x[2] - x[1]**2 / (4 * x[0])]),
                                             "jac": lambda x: np.array([[x[1]**2 / (4* x[0]**2), 
                                                                        -x[1]/(2*x[0]), 
                                                                        1]]),
                                             }
                                p0 = [1, 1, 1]
                                bounds = [(None, None), (None, None), (0, None)]
                                res = minimize(error_func, p0, 
                                               method="SLSQP", 
                                               constraints=[ineq_cons],
                                               bounds=bounds,
                                               options={'ftol': 1e-9, "disp":False},
                                               )
                                param_popt = res.x

                            elif functions[i] == "poly1": 
                                # This special case is necessary as for poly1
                                # negative parameters must be allowed to comply
                                # with ESSC.py
                                # https://github.com/WEC-Sim/WDRT/blob/master/WDRT/ESSC.py
                                if do_use_weights_for_dependence_function:
                                    param_popt, param_pcov = curve_fit(
                                    Fit._get_function(functions[i]),
                                    interval_centers, fit_points,
                                        sigma=fit_points)
                                else:
                                    param_popt, param_pcov = curve_fit(
                                    Fit._get_function(functions[i]),
                                    interval_centers, fit_points,)
                                
                            else:
                                if do_use_weights_for_dependence_function:
                                    param_popt, param_pcov = curve_fit(
                                    Fit._get_function(functions[i]),
                                    interval_centers, fit_points,
                                        sigma=fit_points, bounds=bounds)
                                else:
                                    param_popt, param_pcov = curve_fit(
                                    Fit._get_function(functions[i]),
                                    interval_centers, fit_points, bounds=bounds)

                        except RuntimeError:
                            # Case that optimal parameters not found
                            if i == 0 and name == LOGNORMAL_MU_PARAMETER_KEYWORD:
                                param_name = "sigma"
                            elif i == 2 and name == LOGNORMAL_MU_PARAMETER_KEYWORD:
                                param_name = "mu"
                            elif i == 0:
                                param_name = SHAPE_STRING
                            elif i == 1:
                                param_name = LOCATION_STRING
                            elif i == 2:
                                param_name = SCALE_STRING
                            elif i == 3:
                                param_name = SHAPE2_STRING

                            warnings.warn(
                                "Optimal Parameters not found for parameter '{}' in dimension "
                                "'{}'. Maybe switch the given function for a better fit. Trying "
                                "again with a higher number of calls to function '{}'.".format(
                                    param_name, dimension, functions[i]),
                                RuntimeWarning, stacklevel=2)
                            try:
                                param_popt, param_pcov = curve_fit(
                                    Fit._get_function(functions[i]), interval_centers, fit_points,
                                    bounds=_bounds, maxfev=int(1e6))
                            except RuntimeError:
                                raise RuntimeError(
                                    "Can't fit curve for parameter '{}' in dimension '{}'. "
                                    "Number of iterations exceeded.".format(param_name, dimension))

                        # Save parameter
                        if functions[i] == "alpha3":
                            a = param_popt[0]
                            b = param_popt[1]
                            c = param_popt[2]
                            params[i] = FunctionParam(functions[i], a, b, c,
                                                     C1=C1, C2=C2, C3=C3, C4=C4)
                        elif functions[i] == "poly1":
                            a = param_popt[0]
                            b = param_popt[1]
                            params[i] = FunctionParam(functions[i], a, b, None)
                        else:
                            params[i] = FunctionParam(functions[i], *param_popt)


        # Return particular distribution
        distribution = None
        if name == WEIBULL_2P_KEYWORD or name == WEIBULL_3P_KEYWORD or \
                        name == WEIBULL_3P_KEYWORD_ALTERNATIVE:
            distribution = WeibullDistribution(*params[:3])
        elif name == WEIBULL_EXP_KEYWORD:
            distribution = ExponentiatedWeibullDistribution(*params)
        elif name == LOGNORMAL_MU_PARAMETER_KEYWORD:
            distribution = LognormalDistribution(sigma=params[0], mu=params[2])
        elif name == LOGNORMAL_EXPMU_PARAMETER_KEYWORD:
            distribution = LognormalDistribution(*params[:3])
        elif name == NORMAL_KEYWORD:
            distribution = NormalDistribution(*params[:3])
        elif name == INVERSE_GAUSSIAN_KEYWORD:
            distribution = InverseGaussianDistribution(*params[:3])
        return distribution, dependency, used_number_of_intervals, fit_inspection_data

    def __str__(self):
        return "Fit() instance with dist_dscriptions: " + "".join(
            [str(d) for d in self.dist_descriptions])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Fit data by creating a Fit object
