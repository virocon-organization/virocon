import unittest
import os
import numpy as np
import matplotlib.pyplot as plt

from viroconcom.read_write import read_dataset, read_contour, write_contour
from viroconcom.plot import (plot_contour, plot_wave_breaking_limit, \
    plot_marginal_fit, plot_dependence_functions, plot_confidence_interval, \
    PlottedSample)
from viroconcom.contour_statistics import (points_outside, \
    sort_points_to_form_continous_line)

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (LognormalDistribution, WeibullDistribution, \
    MultivariateDistribution)
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour, HighestDensityContour


class ReadWriteTest(unittest.TestCase):

    def test_read_dataset(self):
        """
        Reads the provided dataset.
        """
        sample_hs, sample_tz, label_hs, label_tz = read_dataset()
        self.assertAlmostEqual(sample_hs[0], 0.2845, delta=0.00001)

    def test_read_write_contour(self):
        """
        Read a contour, then writes this contour to a new file.
        """
        folder_name = 'contour-coordinates/'
        file_name_median = 'doe_john_years_25_median.txt'
        (contour_v_median, contour_hs_median) = read_contour(
            folder_name + file_name_median)
        new_file_path = folder_name + 'test_contour.txt'
        write_contour(contour_v_median, contour_hs_median, new_file_path,
                      'Wind speed (m/s)', 'Significant wave height (m)')
        os.remove(new_file_path)


class ContourStatisticsTest(unittest.TestCase):

    def test_sort_coordinates(self):
        """
        Sorts the points of a contour and plots them.
        """

        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        # Define parameters.
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Compute highest density contours with return periods of 1 and 20 years.
        return_period_1 = 1
        ts = 1  # Sea state duration in hours.
        limits = [(0, 20), (0, 20)]  # Limits of the computational domain.
        deltas = [0.5, 0.5]  # Dimensions of the grid cells.
        hdc = HighestDensityContour(mul_dist, return_period_1, ts, limits, deltas)
        c_unsorted = hdc.coordinates

        # Sort the coordinates.
        c_sorted = sort_points_to_form_continous_line(c_unsorted[0][0],
                                               c_unsorted[0][1],
                                               do_search_for_optimal_start=True)

        # Plot the sorted and unsorted contours.
        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax1 = fig.add_subplot(121)
        plot_contour(x=c_unsorted[0][0],
                     y=c_unsorted[0][1],
                     ax=ax1,
                     contour_label=str(return_period_1) + '-yr contour',
                     line_style='b-')
        ax1.title.set_text('Unsorted')
        ax2 = fig.add_subplot(122)
        plot_contour(x=c_sorted[0],
                     y=c_sorted[1],
                     ax=ax2,
                     contour_label=str(return_period_1) + '-yr contour',
                     line_style='b-')
        ax2.title.set_text('Sorted')
        #plt.show()


class PlottingTest(unittest.TestCase):

    def test_plot_contour_without_sample(self):
        """
        Plots a contour in the most basic way.
        """

        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        # Define parameters.
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_IForm = IFormContour(mul_dist, 50, 3, 50)
        contour_hs = test_contour_IForm.coordinates[0][0]
        contour_tz = test_contour_IForm.coordinates[0][1]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plot_contour(contour_hs, contour_tz, ax)
        #plt.show()

        x_plot, y_plot = ax.lines[0].get_xydata().T
        self.assertAlmostEqual(y_plot[0], contour_tz[0],
                               delta=0.001)

    def test_plot_contour_and_sample(self):
        """
        Plots a contour together with the dataset that has been used to
        fit a distribution for the contour.
        """

        sample_hs, sample_tz, label_hs, label_tz = read_dataset()

        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_hs = {'name': 'Weibull_Exp',
                               'dependency': (None, None, None, None),
                               'width_of_intervals': 0.5}
        dist_description_tz = {'name': 'Lognormal_SigmaMu',
                               'dependency': (0, None, 0),
                               # Shape, Location, Scale
                               'functions': (
                               'asymdecrease3', None, 'lnsquare2'),
                               # Shape, Location, Scale
                               'min_datapoints_for_fit': 50
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz), (
            dist_description_hs, dist_description_tz))

        contour = IFormContour(fit.mul_var_dist, 20, 1, 50)
        contour_hs_20 = contour.coordinates[0][0]
        contour_tz_20 = contour.coordinates[0][1]

        # Find datapoints that exceed the 20-yr contour.
        hs_outside, tz_outside, hs_inside, tz_inside = \
            points_outside(contour_hs_20,
                           contour_tz_20,
                           np.asarray(sample_hs),
                           np.asarray(sample_tz))

        # Compute the median tz conditonal on hs.
        hs = np.linspace(0, 14, 100)
        d1 = fit.mul_var_dist.distributions[1]
        c1 = d1.scale.a
        c2 = d1.scale.b
        tz = c1 + c2 * np.sqrt(np.divide(hs, 9.81))

        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)

        # Plot the 20-year contour and the sample.
        plotted_sample = PlottedSample(x=np.asarray(sample_tz),
                                       y=np.asarray(sample_hs),
                                       ax=ax,
                                       x_inside=tz_inside,
                                       y_inside=hs_inside,
                                       x_outside=tz_outside,
                                       y_outside=hs_outside,
                                       return_period=20)

        plot_contour(x=contour_tz_20,
                     y=contour_hs_20,
                     ax=ax,
                     contour_label='20-yr IFORM contour',
                     x_label=label_tz,
                     y_label=label_hs,
                     line_style='b-',
                     plotted_sample=plotted_sample,
                     x_lim=(0, 19),
                     upper_ylim=15,
                     median_x=tz,
                     median_y=hs,
                     median_label='median of $T_z | H_s$')
        plot_wave_breaking_limit(ax)
        #plt.show()

    def test_plot_seastate_fit(self):
        """
        Plots goodness of fit graphs, for the marginal distribution of X1 and
        for the dependence function of X2|X1. Uses sea state data.

        """

        sample_hs, sample_tz, label_hs, label_tz = read_dataset()

        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_hs = {'name': 'Weibull_Exp',
                               'dependency': (None, None, None, None),
                               'width_of_intervals': 0.5}
        dist_description_tz = {'name': 'Lognormal_SigmaMu',
                               'dependency': (0, None, 0),
                               # Shape, Location, Scale
                               'functions': (
                               'asymdecrease3', None, 'lnsquare2'),
                               # Shape, Location, Scale
                               'min_datapoints_for_fit': 50
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz), (
            dist_description_hs, dist_description_tz))
        dist0 = fit.mul_var_dist.distributions[0]

        fig = plt.figure(figsize=(12.5, 3.5), dpi=150)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        plot_marginal_fit(sample_hs, dist0, fig=fig, ax=ax1, label='$h_s$ (m)',
                          dataset_char='A')
        plot_dependence_functions(fit=fit, fig=fig, ax1=ax2, ax2=ax3,
                                  unconditonal_variable_label=label_hs)
        #plt.show()

    def test_plot_windwave_fit(self):
        """
        Plots goodness of fit graphs, for the marginal distribution of X1 and
        for the dependence function of X2|X1. Uses wind and wave data.
        """

        sample_v, sample_hs, label_v, label_hs = \
            read_dataset('datasets/1year_dataset_D.txt')
        label_v = 'v (m s$^{-1}$)'

        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                               'fixed_parameters': (None, None, None, 5),
                               # shape, location, scale, shape2
                               'dependency': (0, None, 0, None),
                               # shape, location, scale, shape2
                               'functions': (
                               'logistics4', None, 'alpha3', None),
                               # shape, location, scale, shape2
                               'min_datapoints_for_fit': 50,
                               'do_use_weights_for_dependence_function': True}

        # Fit the model to the data.
        fit = Fit((sample_v, sample_hs),
                  (dist_description_v, dist_description_hs))
        dist0 = fit.mul_var_dist.distributions[0]

        fig = plt.figure(figsize=(12.5, 3.5), dpi=150)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        plot_marginal_fit(sample_v, dist0, fig=fig, ax=ax1, label=label_v,
                          dataset_char='D')
        plot_dependence_functions(fit=fit, fig=fig, ax1=ax2, ax2=ax3,
                                  unconditonal_variable_label=label_v)
        #plt.show()

    def test_plot_confidence_interval(self):
        """
        Plots a contour's confidence interval.
        """
        dataset_d_v, dataset_d_hs, label_v, label_hs = \
            read_dataset('datasets/1year_dataset_D.txt')

        # Read the contours that have beem computed previously from csv files.
        folder_name = 'contour-coordinates/'
        file_name_median = 'doe_john_years_25_median.txt'
        file_name_bottom = 'doe_john_years_25_bottom.txt'
        file_name_upper =  'doe_john_years_25_upper.txt'
        (contour_v_median, contour_hs_median) = read_contour(
            folder_name + file_name_median)
        (contour_v_bottom, contour_hs_bottom) = read_contour(
            folder_name + file_name_bottom)
        (contour_v_upper, contour_hs_upper) = read_contour(
            folder_name + file_name_upper)

        # Plot the sample, the median contour and the confidence interval.
        fig = plt.figure(figsize=(5, 5), dpi=150)
        ax = fig.add_subplot(111)
        plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                                       y=np.asarray(dataset_d_hs),
                                       ax=ax,
                                       label='dataset D')
        contour_labels = ['50th percentile contour',
                          '2.5th percentile contour',
                          '97.5th percentile contour']
        plot_confidence_interval(
            x_median=contour_v_median, y_median=contour_hs_median,
            x_bottom=contour_v_bottom, y_bottom=contour_hs_bottom,
            x_upper=contour_v_upper, y_upper=contour_hs_upper, ax=ax,
            x_label=label_v,
            y_label=label_hs, contour_labels=contour_labels,
            plotted_sample=plotted_sample)
        #plt.show()
