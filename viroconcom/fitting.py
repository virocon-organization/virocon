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
from scipy.optimize import curve_fit

from .settings import SHAPE_STRING, LOCATION_STRING, SCALE_STRING
from .params import ConstantParam, FunctionParam
from .distributions import (WeibullDistribution, LognormalDistribution, NormalDistribution,
                            KernelDensityDistribution, MultivariateDistribution)


__all__ = ["Fit"]


# Functions for fitting
# Power function
def _power3(x, a, b, c):
    return a + b * x ** c


# Exponential function
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)


# Bounds for function parameters
# 0 < a < inf
# 0 < b < inf
# -inf < c < inf
_bounds = ([np.finfo(np.float64).tiny, np.finfo(np.float64).tiny, -np.inf],
          [np.inf, np.inf, np.inf])


class BasicFit():
    """
    Holds the parameters (shape, loc, scale) and also the raw data to a single fit.

    Attributes
    ----------
    shape : float
        The shape parameter for the fit.

    loc : float
        The location parameter for the fit.

    scale : float
        The scale parameter for the fit.

    samples : list of float
        The raw data that is used for this fit. For that case that there is no dependency this
        list contains the whole data of the dimension.

    """

    def __init__(self, shape, loc, scale, samples):

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

        # Raw data
        self.samples = samples

    def __str__(self):
        return "BasicFit with shape={}, loc={}, scale={}.".format(
            self.shape, self.loc, self.scale)


class FitInspectionData():
    """
    This class holds information for plotting the fits of a single dimension. It is used to give
    a visual look about how good the fits in this dimension were.

    Attributes
    ----------
    used_number_of_intervals : int
        The actually number of intervals this dimension is divided for other dependent dimensions.

    shape_at : list of float
        This list contains the values of the divided dimension the shape parameter depends on.

    shape_value : list of float
        The associated values of the parameter shape to the divided dimension the shape
        parameter depends on.

    loc_at : list of float
        This list contains the values of the divided dimension the location parameter depends on.

    loc_value : list of float
        The associated values of the parameter loc to the divided dimension the location
        parameter depends on.

    scale_at : list of float
        This list contains the values of the divided dimension the scale parameter depends on.

    scale_value : list of float
        The associated values of the parameter scale to the divided dimension the scale
        parameter depends on.

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
        self._shape_value = [[], [], []]

        self.loc_at = None
        self._loc_value = [[], [], []]

        self.scale_at = None
        self._scale_value = [[], [], []]

        # Raw data for each parameter of this dimension
        self.shape_samples = []
        self.loc_samples = []
        self.scale_samples = []

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
            self.shape_samples.append(basic_fit.samples)
        elif param == LOCATION_STRING:
            self._loc_value[0].append(basic_fit.shape)
            self._loc_value[1].append(basic_fit.loc)
            self._loc_value[2].append(basic_fit.scale)
            self.loc_samples.append(basic_fit.samples)
        elif param == SCALE_STRING:
            self._scale_value[0].append(basic_fit.shape)
            self._scale_value[1].append(basic_fit.loc)
            self._scale_value[2].append(basic_fit.scale)
            self.scale_samples.append(basic_fit.samples)
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
            return BasicFit(self._shape_value[0][index], self._shape_value[1][index],
                            self._shape_value[2][index], self.shape_samples[index])
        elif param == LOCATION_STRING:
            return BasicFit(self._loc_value[0][index], self._loc_value[1][index],
                            self._loc_value[2][index], self.loc_samples[index])
        elif param == SCALE_STRING:
            return BasicFit(self._scale_value[0][index], self._scale_value[1][index],
                            self._scale_value[2][index], self.scale_samples[index])
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

    >>> dist_description_0 = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 3}
    >>> dist_description_1 = {'name': 'Lognormal', 'dependency': (None, None, 0), 'functions': (None, None, 'exp3')}
    >>> my_fit = Fit((sample_1, sample_2), (dist_description_0, dist_description_1))
    >>>
    >>> #fig = plt.figure(figsize=(10, 8))
    >>> #example_text = fig.suptitle("Dependence of 'scale'")
    >>>
    >>> #ax_1 = fig.add_subplot(221)
    >>> #title1 = ax_1.set_title("Fitted curve")
    >>> param_grid = my_fit.multiple_fit_inspection_data[1].scale_at
    >>> x_1 = np.linspace(5, 15, 100)
    >>> #ax1_plot = ax_1.plot(param_grid, my_fit.multiple_fit_inspection_data[0].scale_value, 'x')
    >>> #example_plot1 = ax_1.plot(x_1, my_fit.mul_var_dist.distributions[1].scale(x_1))
    >>>
    >>> #ax_2 = fig.add_subplot(222)
    >>> #title2 = ax_2.set_title("Distribution '1'")
    >>> #ax2_hist = ax_2.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[0], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[0])
    >>> #ax2_plot = ax_2.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
    >>>
    >>> #ax_3 = fig.add_subplot(223)
    >>> #title3 = ax_3.set_title("Distribution '2'")
    >>> #ax3_hist = ax_3.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[1], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[1])
    >>> #ax3_plot = ax_3.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
    >>>
    >>> #ax_4 = fig.add_subplot(224)
    >>> #title4 = ax_4.set_title("Distribution '3'")
    >>> #ax4_hist = ax_4.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[2], normed=1)
    >>> shape = my_fit.mul_var_dist.distributions[1].shape(0)
    >>> scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[2])
    >>> #ax4_plot = ax_4.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))

    """

    def __init__(self, samples, dist_descriptions, timeout=None):
        """
        Creates a Fit, by computing the distribution that describes the samples 'best'.

        Parameters
        ----------
        samples : list of list
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
        dist_descriptions contains the following keys:

        name : str
            name of distribution:

            - Weibull,
            - Lognormal (shape, scale),
            - Lognormal_SigmaMu (sigma, mu),
            - Normal,
            - KernelDensity (no dependency)

        dependency : list of int
            Length of 3 in the order (shape, loc, scale) contains:

            - None -> no dependency
            - int -> depends on particular dimension

        functions : list of str
            Length of 3 in the order : (shape, loc, scale), usable options:

            - :power3: :math:`a + b * x^c`
            - :exp3: :math:`a + b * e^{x * c}`
            - remark : in case of Lognormal_SigmaMu it is (sigma, None, mu)

        and either number_of_intervals or width_of_intervals:

        number_of_intervals : int
            Number of bins the data of this variable should be seperated for fits which depend
            upon it. If the number of bins is given, the width of the bins is determined
            automatically.

        width_of_bins : float
            Width of the bins. When the width of the bins is given, the number of bins is
            determined automatically.

        """
        self.dist_descriptions = dist_descriptions # Compute references this attribute at plot.py

        list_number_of_intervals = []
        list_width_of_intervals = []
        for dist_description in dist_descriptions:
            list_number_of_intervals.append(dist_description.get('number_of_intervals'))
            list_width_of_intervals.append(dist_description.get('width_of_intervals'))
        for dist_description in dist_descriptions:
            dist_description['list_number_of_intervals'] = list_number_of_intervals
            dist_description['list_width_of_intervals'] = list_width_of_intervals

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
    def _fit_distribution(sample, name):
        """
        Fits the distribution and returns the parameters.

        Parameters
        ----------
        sample : list of float
            Raw data the distribution is fitted on.
        name : str
            Name of the distribution ("Weibull", "Lognormal" or
            "Lognormal_SigmaMu", "Normal", "KernelDensity").
        Returns
        -------
        tuple of ConstantParam
             The computed parameters in the order of (shape, loc, scale).
        Raises
        ------
        ValueError
            If the distribution is unknown.
        """

        if name == 'Weibull':
            params = sts.weibull_min.fit(sample)
        elif name == 'Normal':
            params = list(sts.norm.fit(sample))
            # Shape doesn't exist for normal
            params.insert(0, 0)
        elif name[:9] == 'Lognormal':
            # For lognormal loc is set to 0
            params = sts.lognorm.fit(sample, floc=0)
        elif name == 'KernelDensity':
            dens = sm.nonparametric.KDEUnivariate(sample)
            dens.fit(gridsize=2000)
            # Kernel density doesn't have shape, loc, scale
            return (dens.cdf, dens.icdf)
        else:
            err_msg = "Distribution '{}' is unknown.".format(name)
            raise ValueError(err_msg)

        return (ConstantParam(params[0]), ConstantParam(params[1]), ConstantParam(params[2]))

    @staticmethod
    def _get_function(function_name):
        """
        Returns the function.

        Parameters
        ----------
        function_name : str
            Options are 'power3', 'exp3'.

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
        elif function_name is None:
            return None
        else:
            err_msg = "Function '{}' is unknown.".format(function_name)
            raise ValueError(err_msg)

    @staticmethod
    def _append_params(name, param_values, dependency, index, sample):
        """
        Distributions are being fitted and the results are appended to param_points.

        Parameters
        ----------
        name : str
            Name of distribution (Weibull, Lognormal, Normal, KernelDensity (no dependency)).
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

        Returns
        -------
        BasicFit
             The information of this single fit.
        """

        # Fit distribution
        current_params = Fit._fit_distribution(sample, name)

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
                            number_of_intervals=None, bin_width=None):
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
            Name of distribution (Weibull, Lognormal, Normal, KernelDensity (no dependency)).
        dependency : list of int
            Length of 3 in the order (shape, loc, scale) contains :
                None -> no dependency
                int -> depends on particular dimension
        index : int
            Order : (shape, loc, scale) (i.e. 0 -> shape).
        number_of_intervals : int
            Number of distributions used to fit shape, loc, scale.
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
            If the parameter number_of_intervals or bin_width was not specified.
        RuntimeError
            If there was not enough data and the number of intervals was less than three.
        """
        MIN_DATA_POINTS_FOR_FIT = 10

        # Compute intervals.
        if number_of_intervals:
            interval_centers, interval_width = np.linspace(
                min(samples[dependency[index]]), max(samples[dependency[index]]),
                num=number_of_intervals, endpoint=False, retstep=True)
            interval_centers += 0.5 * interval_width
        elif bin_width:
            interval_width = bin_width
            interval_centers = np.arange(
                0.5 * interval_width,
                max(samples[dependency[index]]) + 0.5 * interval_width,
                interval_width)
        else:
            raise RuntimeError(
                "Either the parameters number_of_intervals or bin_width has to be specified, "
                "otherwise the intervals are not specified. Exiting.")

        # Sort samples.
        samples = np.stack((sample, samples[dependency[index]])).T
        sort_indice = np.argsort(samples[:, 1])
        sorted_samples = samples[sort_indice]

        # Return values.
        param_values = [[], [], []]
        dist_values = []

        # List of all basic fits.
        multiple_basic_fit = []

        # Deleted interval_centers by index.
        deleted_centers = []

        # Define the data interval that is used for the fit.
        for i, step in enumerate(interval_centers):
            mask = ((sorted_samples[:, 1] >= step - 0.5 * interval_width) &
                    (sorted_samples[:, 1] < step + 0.5 * interval_width))
            samples_in_interval = sorted_samples[mask, 0]
            if len(samples_in_interval) >= MIN_DATA_POINTS_FOR_FIT:
                try:
                    # Fit distribution to selected data.
                    basic_fit = Fit._append_params(
                        name, param_values, dependency, index, samples_in_interval)
                    multiple_basic_fit.append(basic_fit)
                    dist_values.append(samples_in_interval)
                except ValueError:
                    # For case that to few fitting data for the step were found
                    # the step is deleted.
                    deleted_centers.append(i) # Add index of unused center.
                    warnings.warn(
                        "There is not enough data for step '{}' in dimension "
                        "'{}'. This step is skipped. Consider analyzing your "
                        "data or reducing the number of intervals."
                            .format(step, dependency[index]),
                        RuntimeWarning, stacklevel=2)
            else:
                # For case that to few fitting data for the step were found
                # the step is deleted.
                deleted_centers.append(i) # Add index of unused center.
                warnings.warn(
                    "'Due to the restriction of MIN_DATA_POINTS_FOR_FIT='{}' "
                    "there is not enough data (n='{}') for the interval "
                    "centered at '{}' in dimension '{}'. This step is skipped. "
                    "Consider analyzing your data or reducing the number of "
                    "intervals."
                        .format(MIN_DATA_POINTS_FOR_FIT,
                        len(samples_in_interval),
                        step,
                        dependency[index]),
                    RuntimeWarning, stacklevel=2)
        if len(interval_centers) < 3:
            nr_of_intervals = str(len(interval_centers))
            raise RuntimeError("Your settings resulted in " + nr_of_intervals +
                               " intervals. However, at least 3 intervals are "
                               "required. Consider changing the interval width "
                               "setting.")

        # Delete interval centers that were not used.
        interval_centers = np.delete(interval_centers, deleted_centers)

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
        name = kwargs.get('name', 'Weibull')
        dependency = kwargs.get('dependency', (None, None, None))
        functions = kwargs.get('functions', ('polynomial', 'polynomial', 'polynomial'))
        list_number_of_intervals = kwargs.get('list_number_of_intervals')
        list_width_of_intervals = kwargs.get('list_width_of_intervals')

        # Fit inspection data for current dimension
        fit_inspection_data = FitInspectionData()

        # Initialize used_number_of_intervals (shape, loc, scale
        used_number_of_intervals = [None, None, None]

        # Handle KernelDensity separated
        if name == 'KernelDensity':
            if dependency != (None, None, None):
                raise NotImplementedError("KernelDensity can not be conditional.")
            return KernelDensityDistribution(Fit._fit_distribution(sample, name)), dependency, \
                   used_number_of_intervals, fit_inspection_data

        # Initialize params (shape, loc, scale)
        params = [None, None, None]

        for index in range(len(dependency)):

            # Continue if params is yet computed
            if params[index] is not None:
                continue

            # In case that there is no dependency for this param
            if dependency[index] is None:
                current_params = Fit._fit_distribution(sample, name)

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

                        if i == 2 and name == 'Lognormal_SigmaMu':
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
                            number_of_intervals=list_number_of_intervals[dependency[index]])
                # If a the (constant) width of the intervals is given.
                elif list_width_of_intervals[dependency[index]]:
                    interval_centers, dist_values, param_values, multiple_basic_fit = \
                        Fit._get_fitting_values(
                            sample, samples, name, dependency, index,
                            bin_width=list_width_of_intervals[dependency[index]])

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

                        # Add interval centers to fit inspection data
                        if i == 0:
                            fit_inspection_data.shape_at = interval_centers
                        elif i == 1:
                            fit_inspection_data.loc_at = interval_centers
                        elif i == 2:
                            fit_inspection_data.scale_at = interval_centers

                        # Add used number of intervals for current parameter
                        used_number_of_intervals[i] = len(interval_centers)

                        if i == 2 and name == 'Lognormal_SigmaMu':
                            fit_points = [np.log(p(None)) for p in param_values[i]]
                        else:
                            fit_points = [p(None) for p in param_values[i]]
                        # Fit parameters with particular function
                        try:
                            param_popt, param_pcov = curve_fit(
                                Fit._get_function(functions[i]),
                                interval_centers, fit_points, bounds=_bounds)
                        except RuntimeError:
                            # Case that optimal parameters not found
                            if i == 0 and name == 'Lognormal_SigmaMu':
                                param_name = "sigma"
                            elif i == 2 and name == 'Lognormal_SigmaMu':
                                param_name = "mu"
                            elif i == 0:
                                param_name = SHAPE_STRING
                            elif i == 1:
                                param_name = LOCATION_STRING
                            elif i == 2:
                                param_name = SCALE_STRING

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
                        params[i] = FunctionParam(*param_popt, functions[i])

        # Return particular distribution
        distribution = None
        if name == 'Weibull':
            distribution = WeibullDistribution(*params)
        elif name == 'Lognormal_SigmaMu':
            distribution = LognormalDistribution(sigma=params[0], mu=params[2])
        elif name == 'Lognormal':
            distribution = LognormalDistribution(*params)
        elif name == 'Normal':
            distribution = NormalDistribution(*params)
        return distribution, dependency, used_number_of_intervals, fit_inspection_data

    def __str__(self):
        return "Fit() instance with dist_dscriptions: " + "".join(
            [str(d) for d in self.dist_descriptions])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Fit data by creating a Fit object
