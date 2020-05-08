import unittest
import numpy as np
import matplotlib.pyplot as plt

from viroconcom.read_write import read_dataset
from viroconcom.plot import plot_contour, plot_marginal_fit, \
    plot_dependence_functions, PlottedSample

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import LognormalDistribution, WeibullDistribution, \
    MultivariateDistribution
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from viroconcom.contour_statistics import points_outside


class ReadWriteTest(unittest.TestCase):

    def test_read_dataset(self):
        """
        Reads the provided dataset.
        """
        sample_hs, sample_tz, label_hs, label_tz = read_dataset()
        self.assertAlmostEqual(sample_hs[0], 0.2845, delta=0.00001)


class PlottingTest(unittest.TestCase):

    def plot_contour_without_sample(self):

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

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plot_contour(test_contour_IForm.coordinates[0],
                     test_contour_IForm.coordinates[1], ax)

    def plot_contour_and_sample(self):
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
        contour_hs_20 = contour.coordinates[0]
        contour_tz_20 = contour.coordinates[1]

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

    def plot_fit(self):
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
        plot_dependence_functions(fit=fit, fig=fig, ax1=ax2, ax2=ax3, unconditonal_variable_label=label_hs)