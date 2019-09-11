#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test class for module PCA

Author:  mish-mosh
-------

"""

import unittest
import os

import numpy as np
import pandas as pd

from .context import viroconcom
from viroconcom.PCA import PCAFit

from pandas.util.testing import assert_numpy_array_equal

_here = os.path.dirname(__file__)
testFiles_path = os.path.abspath(os.path.join(_here, "testfiles"))


class PCATest(unittest.TestCase):

    def test_get_fitted_contour(self):
        """
        Tests the method get_fitted_contour (see below)
        """

        # test dataFrame
        test_df = pd.read_csv('https://raw.githubusercontent.com/ec-benchmark-organizers/ec-benchmark/'
                              'master/datasets/A.txt', sep=';', usecols=[1, 2])
        # test PCA
        test_pca = PCAFit(test_df)

        ##  generating the contour
        from viroconcom.fitting import Fit
        from viroconcom.contours import IFormContour

        dist_description_zup = {'name': 'Lognormal_SigmaMu', 'sigma': (0.00, 0.308, -0.250), 'mu': (1.47, 0.214, 0.641)}
        dist_description_swh = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 2}

        # columns to plot and fit
        col1 = test_df.values.transpose()[1].tolist()
        col2 = test_df.values.transpose()[0].tolist()

        # fitting
        my_fit = Fit((col1, col2), (dist_description_zup, dist_description_swh))

        # iform contour
        iform_contour_original = IFormContour(my_fit.mul_var_dist, 10, 5, 500)

        # test contour
        test_contour = pd.DataFrame(np.vstack(test_pca.get_fitted_contour(iform_contour_original.coordinates[0])).
                                    transpose())

        # comparison contour
        compare_contour = pd.read_csv(testFiles_path + "/DataSet_A_contour_PCA_fitted.csv", dtype='str')
        compare_contour = compare_contour.astype('float64')

        # assersion:
        assert_numpy_array_equal(test_contour.values, compare_contour.values)


if __name__ == '__main__':
    unittest.main()
