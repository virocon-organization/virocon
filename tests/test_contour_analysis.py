#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests the module contour_analysis.
"""

import unittest
import matplotlib.pyplot as plt

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.contour_analysis import points_outside
from viroconcom.plot import plot_contour


class ContourAnalysisTest(unittest.TestCase):

    def tests_points_outside(self):
        """
        Tests whether points_outside identifies the correct number of points
        outside the contour.
        """
        # Load a metocean dataset an environmental contour.
        v, hs, label_v, label_hs = read_ecbenchmark_dataset('datasets/1year_dataset_D.txt')
        folder_name = 'contour-coordinates/'
        file_name_median = 'doe_john_years_25_median.txt'
        (contour_v, contour_hs) = read_contour(folder_name + file_name_median)

        # Compute the points outside the contour.
        (v_outside, hs_outside, v_inside, hs_inside) = points_outside(contour_v, contour_hs, v, hs)

        # Manually check the results, 5 points are outside.
        fig, ax = plt.subplots()
        ax.scatter(v_inside, hs_inside)
        ax.scatter(v_outside, hs_outside, c='r')
        plot_contour(contour_v, contour_hs, ax=ax, x_label=label_v, y_label=label_hs)
        #plt.show()

        self.assertEqual(len(v_outside), 5)
