import unittest
import matplotlib.pyplot as plt

from viroconcom.plot import plot_contour
from viroconcom.contours import sort_points_to_form_continous_line

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (LognormalDistribution, WeibullDistribution, \
    MultivariateDistribution)
from viroconcom.contours import HighestDensityContour

# This is commented out as it does not ensure that the sorting algorithm
# functions as intended.

#class ContourStatisticsTest(unittest.TestCase):
#
#
    # def test_sort_coordinates(self):
    #     """
    #     Sorts the points of a highest density contour and plots them.
    #     """
    #
    #     # Define dependency tuple.
    #     dep1 = (None, None, None)
    #     dep2 = (0, None, 0)
    #
    #     # Define parameters.
    #     shape = ConstantParam(1.471)
    #     loc = ConstantParam(0.8888)
    #     scale = ConstantParam(2.776)
    #     par1 = (shape, loc, scale)
    #
    #     mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
    #     sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)
    #
    #     # Create distributions.
    #     dist1 = WeibullDistribution(*par1)
    #     dist2 = LognormalDistribution(mu=mu, sigma=sigma)
    #
    #     distributions = [dist1, dist2]
    #     dependencies = [dep1, dep2]
    #
    #     mul_dist = MultivariateDistribution(distributions, dependencies)
    #
    #     # Compute highest density contours with return periods of 1 and 20 years.
    #     return_period_1 = 1
    #     ts = 1  # Sea state duration in hours.
    #     limits = [(0, 20), (0, 20)]  # Limits of the computational domain.
    #     deltas = [0.5, 0.5]  # Dimensions of the grid cells.
    #     hdc = HighestDensityContour(mul_dist, return_period_1, ts, limits, deltas)
    #     c_unsorted = hdc.coordinates
    #
    #     # Sort the coordinates.
    #     c_sorted = sort_points_to_form_continous_line(c_unsorted[0][0],
    #                                            c_unsorted[0][1],
    #                                            do_search_for_optimal_start=True)
    #
    #     # Plot the sorted and unsorted contours.
    #     fig = plt.figure(figsize=(10, 5), dpi=150)
    #     ax1 = fig.add_subplot(121)
    #     plot_contour(x=c_unsorted[0][0],
    #                  y=c_unsorted[0][1],
    #                  ax=ax1,
    #                  contour_label=str(return_period_1) + '-yr contour',
    #                  line_style='b-')
    #     ax1.title.set_text('Unsorted')
    #     ax2 = fig.add_subplot(122)
    #     plot_contour(x=c_sorted[0],
    #                  y=c_sorted[1],
    #                  ax=ax2,
    #                  contour_label=str(return_period_1) + '-yr contour',
    #                  line_style='b-')
    #     ax2.title.set_text('Sorted')
    #     #plt.show()