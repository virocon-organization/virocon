#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests the computation of contours.
"""

import unittest
import os

import numpy as np
import pandas as pd

from .context import viroconcom

from viroconcom.params import ConstantParam, FunctionParam

from viroconcom.distributions import (
    WeibullDistribution, ExponentiatedWeibullDistribution, LognormalDistribution,
    NormalDistribution, MultivariateDistribution)
from viroconcom.contours import IFormContour, ISormContour, \
    HighestDensityContour, DirectSamplingContour


_here = os.path.dirname(__file__)
testfiles_path = os.path.abspath(os.path.join(_here, "testfiles"))


class ContourCreationTest(unittest.TestCase):

    def test_HDC2d_WL(self):
        """
        2-d HDC with Weibull and Lognormal distribution.

        The used probabilistic model is described in Vanem and Bitner-Gregersen
        (2012), DOI: 10.1016/j.apor.2012.05.006
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)

        #del shape, loc, scale

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        #del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
        #calc contour
        n_years = 50
        limits = [(0, 20), (0, 18)]
        deltas = [0.1, 0.1]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        finaldt0 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0],
                                'y' : test_contour_HDC.coordinates[1]})


        result0 = pd.read_csv(testfiles_path + "/HDC2dWL_coordinates.csv")

        for g,h in [(g, h) for g in result0.index for h in result0.columns]:
            self.assertAlmostEqual(result0.loc[g, h], finaldt0.loc[g, h], places=8)


    def test_HDC2d_ExponentiatedWbl(self):
        """
        2-d HDC with exponentiated Weibull distributions.
        """

        # Define dependency tuple.
        dep1 = (None, None, None, None) # shape, location, scale, shape2
        dep2 = (None, None, 0, None) # shape, location, scale, shape2

        # Define parameters.
        v_shape = ConstantParam(11)
        v_loc = None
        v_scale = ConstantParam(2.6)
        v_shape2 = ConstantParam(0.54)
        par1 = (v_shape, v_loc, v_scale, v_shape2)

        hs_shape = ConstantParam(1.4)
        hs_loc = None
        hs_scale = FunctionParam('power3', 0.15, 0.0033, 2.45)
        hs_shape2 = ConstantParam(5)
        par2 = (hs_shape, hs_loc, hs_scale, hs_shape2)

        # Create distributions.
        dist1 = ExponentiatedWeibullDistribution(*par1)
        dist2 = ExponentiatedWeibullDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Calculate the contour.
        n_years = 50
        limits = [(0, 20), (0, 18)]
        deltas = [0.1, 0.1]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        # If we knew the correct coordinates we could continue with something like
        # this:
        #contour_coordinates = pd.DataFrame({'x' : test_contour_HDC.coordinates[0][0],
        #                        'y' : test_contour_HDC.coordinates[0][1]})
        #result0 = pd.read_csv(testfiles_path + "/filename.csv")
        #for g,h in [(g, h) for g in result0.index for h in result0.columns]:
        #    self.assertAlmostEqual(result0.loc[g, h], contour_coordinates.loc[g, h], places=8)


    def test_omae2020_wind_wave_contour(self):
        """
        Contour similar to the wind-wave contour in 'Global hierararchical models
        for wind and wave contours', dataset D. First variable = wind speed,
        second variable = significant wave height.
        """

        # Define dependency tuple.
        dep1 = (None, None, None, None) # shape, location, scale, shape2
        dep2 = (0, None, 0, None) # shape, location, scale, shape2

        # Define parameters.
        v_shape = ConstantParam(2.42)
        v_loc = None
        v_scale = ConstantParam(10)
        v_shape2 = ConstantParam(0.761)
        par1 = (v_shape, v_loc, v_scale, v_shape2)

        hs_shape = FunctionParam('logistics4', 0.582, 1.90, 0.248, 8.49)
        hs_loc = None
        hs_scale = FunctionParam('alpha3', 0.394, 0.0178, 1.88,
                                 C1=0.582, C2=1.90, C3=0.248, C4=8.49)

        hs_shape2 = ConstantParam(5)
        par2 = (hs_shape, hs_loc, hs_scale, hs_shape2)

        # Create distributions.
        dist1 = ExponentiatedWeibullDistribution(*par1)
        dist2 = ExponentiatedWeibullDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Calculate the contour.
        n_years = 50
        limits = [(0, 40), (0, 20)]
        deltas = [0.1, 0.1]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 1,
                                                 limits, deltas)

        # Compare the computed contours to the contours published in
        # 'Global hierarchical models for wind and wave contours', Figure 8.
        max_v = max(test_contour_HDC.coordinates[0])
        self.assertAlmostEqual(max_v, 29.5, delta=0.5) # Should be about 29.5
        max_hs = max(test_contour_HDC.coordinates[1])
        self.assertAlmostEqual(max_hs, 14.5, delta=0.5) # Should be about 15

        # Now calculate the same contour without defining the grid.
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 1)

        # Compare the computed contours to the contours published in
        # 'Global hierarchical models for wind and wave contours', Figure 8.
        max_v = max(test_contour_HDC.coordinates[0])
        self.assertAlmostEqual(max_v, 29.5, delta=0.5) # Should be about 29.5
        max_hs = max(test_contour_HDC.coordinates[1])
        self.assertAlmostEqual(max_hs, 14.5, delta=0.5) # Should be about 15


    def test_HDC3d_WLL(self):
        """
        Contour example for 3-d HDC with Weibull, Lognormal and
        Lognormal distribution.
        """

        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)


        #del shape, loc, scale

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)
        dist3 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2, dist3]
        dependencies = [dep1, dep2, dep3]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        #del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
        #calc contour
        n_years = 50
        limits = [(0, 20), (0, 18),(0, 18)]
        deltas = [1, 1, 1]

        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        finaldt = pd.DataFrame({'x' : test_contour_HDC.coordinates[0],
                                'y' : test_contour_HDC.coordinates[1],
                                'z' : test_contour_HDC.coordinates[2]})


        result = pd.read_csv(testfiles_path + "/HDC3dWLL_coordinates.csv")
        for i,j in [(i, j) for i in result.index for j in result.columns]:
            self.assertAlmostEqual(result.loc[i,j], finaldt.loc[i,j], places=8)


    def test_HDC4d_WLLL(self):
        """
        Contour example for a 4-dimensinal HDC with Weibull, Lognormal,
        Lognormal and Lognormal distribution.
        """

        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)
        dep4 = (0, None, 0)

        # Define parameters.
        shape = ConstantParam(2.776)
        loc = ConstantParam(1.471)
        scale = ConstantParam(0.8888)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)
        dist3 = LognormalDistribution(mu=mu, sigma=sigma)
        dist4 = LognormalDistribution(mu=mu, sigma=sigma)


        distributions = [dist1, dist2, dist3, dist4]
        dependencies = [dep1, dep2, dep3, dep4]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Compute contour.
        n_years = 50
        limits = [(0, 20), (0, 18), (0, 18), (0, 18)]
        deltas = [1, 1, 1, 1]

        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)



    def test_HDC2d_WN(self):
        """
        Creating a contour example.
        """


        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (None, 0, 0)

        # Define parameters.
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam('power3', 4, 10, 0.02)
        scale = FunctionParam('exp3', 0.1, 0.02, -0.1)
        par2 = (shape, loc, scale)

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = NormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Compute the contour.
        n_years = 50
        limits = [(0, 20), (0, 20)]
        deltas = [0.05, 0.01]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        finaldt2 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0],
                                 'y' : test_contour_HDC.coordinates[1]})

        result2 = pd.read_csv(testfiles_path + "/HDC2dWN_coordinates.csv")

        for k,l in [(k, l) for k in result2.index for l in result2.columns]:
            self.assertAlmostEqual(result2.loc[k,l], finaldt2.loc[k,l], places=8)



    def test_HDC3d_WLN(self):

        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (None, 0, 0)

        # Define parameters.
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam('power3', 4, 10, 0.02)
        scale = FunctionParam('exp3', 0.1, 0.02, -0.1)
        par2 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1, 1.5, 0.2)
        sigma = FunctionParam('exp3', 0.1, 0.2, -0.2)

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)
        dist3 = NormalDistribution(*par2)

        distributions = [dist1, dist2, dist3]
        dependencies = [dep1, dep2, dep3]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        del mu, sigma
        #del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
        #calc contour
        n_years = 50
        limits = [(0, 20), (0, 20),(0, 20)]
        deltas = [0.5, 0.5, 0.05]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        finaldt3 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0],
                                 'y' : test_contour_HDC.coordinates[1],
                                 'z' : test_contour_HDC.coordinates[2]})

        matlab3 = pd.read_csv(testfiles_path + "/hdc3d_wln.csv", names=['x', 'y', 'z'])

        result3 = pd.read_csv(testfiles_path + "/HDC3dWLN_coordinates.csv")
        for m,n in [(m, n) for m in result3.index for n in result3.columns]:
            self.assertAlmostEqual(result3.loc[m, n], finaldt3.loc[m, n], places=8)

    def test_IForm2d_WL(self):
        """
        2-d IFORM contour.

        The used probabilistic model is described in Vanem and Bitner-Gregersen
        (2012), DOI: 10.1016/j.apor.2012.05.006 .
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

        calculated_coordinates = pd.DataFrame({'x' : test_contour_IForm.coordinates[0],
                                 'y' : test_contour_IForm.coordinates[1]})
        #calculated_coordinates.to_csv('save_this_file.csv', sep=',', header=['x', 'y'], index=False)

        true_coordinates = pd.read_csv(testfiles_path + "/IForm2dWL_coordinates.csv")
        for o,p in [(o, p) for o in true_coordinates.index for p in true_coordinates.columns]:
            self.assertAlmostEqual(calculated_coordinates.loc[o, p], true_coordinates.loc[o, p], places=8)

    def test_IForm2d_WN(self):
        """
        2-d IFORM contour.
        """

        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (None, 0, 0)

        # Define parameters.
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam('power3', 7, 1.489, 0.1901)
        scale = FunctionParam('exp3', 1.5, 0.1748, -0.2243)
        par2 = (shape, loc, scale)

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = NormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_IForm = IFormContour(mul_dist, 50, 3, 50)

        calculated_coordinates = pd.DataFrame({'x' : test_contour_IForm.coordinates[0],
                                 'y' : test_contour_IForm.coordinates[1]})

        true_coordinates = pd.read_csv(testfiles_path + "/IForm2dWN_coordinates.csv")

        for r,s in [(r, s) for r in true_coordinates.index for s in true_coordinates.columns]:
          self.assertAlmostEqual(calculated_coordinates.loc[r, s], true_coordinates.loc[r, s], places=8)

    def test_IForm3d(self): # TODO what does this test do
        """
        3-dimensional IFORM contour.
        """

        # Define dependency tuple.
        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)

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
        dist3 = LognormalDistribution(mu=mu, sigma=sigma)
        distributions = [dist1, dist2, dist3]
        dependencies = [dep1, dep2, dep3]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_iform = IFormContour(mul_dist, 50, 3, 400)

    def test_isorm2d_WL(self):
        """
        ISORM contour with Vanem2012 model.

        The used probabilistic model is described in Vanem and Bitner-Gregersen
        (2012), DOI: 10.1016/j.apor.2012.05.006
        """

        # Define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        # Define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
        sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)

        # Create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_isorm = ISormContour(mul_dist, 50, 3, 50)

        calculated_coordinates = pd.DataFrame({'x': test_contour_isorm.coordinates[0],
                                               'y': test_contour_isorm.coordinates[1]})

        true_coordinates = pd.read_csv(testfiles_path + "/isorm2dWL_coordinates.csv")
        for o, p in [(o, p) for o in true_coordinates.index for p in true_coordinates.columns]:
            self.assertAlmostEqual(calculated_coordinates.loc[o, p], true_coordinates.loc[o, p], places=8)


class HDCTest(unittest.TestCase):

    def _setup(self,
               limits=[(0, 20), (0, 20)],
               deltas=[0.05, 0.05],
               n_years = 25,
               dep1=(None, None, None),
               dep2=(0, None, 0),
               par1=(ConstantParam(1.471), ConstantParam(0.8888),
                     ConstantParam(2.776)),
               par2=(FunctionParam('exp3', 0.0400, 0.1748, -0.2243), None,
                     FunctionParam('power3', 0.1, 1.489, 0.1901))
               ):
        """
        Creating a contour (same as in DOI: 10.1016/j.coastaleng.2017.03.002).
        """

        self.limits = limits
        self.deltas = deltas
        self.n_years = n_years

        # Define dependency tuple.
        self.dep1 = dep1
        self.dep2 = dep2

        # Define parameters.
        self.par1 = par1
        self.par2 = par2

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Compute contour.
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)
        return test_contour_HDC


    def test_cumsum(self):
        """
        Tests if the return values of cumsum_biggest_until are correct.
        """

        test_contour_HDC = self._setup()
        data_example = np.array([[80, 7, 20, 40], [1, 9, 45, 23]])

        summed_fields = test_contour_HDC.cumsum_biggest_until(data_example,
                                                              165.0)[0]
        last_summed = test_contour_HDC.cumsum_biggest_until(data_example,
                                                            165.0)[1]
        np.testing.assert_array_equal(summed_fields,
             ([[1, 0, 0, 1], [0, 0, 1, 0]]),
             'cumsum calculates wrong summed_fields')
        self.assertEqual(last_summed, 40, 'cumsum gives wrong last_summed')


    def test_cumsum_nan_entry(self):
        """
        Tests if ValueError is raised when the array has a 'nan' entry.
        """

        test_contour_HDC = self._setup()
        data_example_nan = np.array([[80, 7, float('nan'), 40], [1, 9, 45, 23]])
        with self.assertRaises(ValueError):
            test_contour_HDC.cumsum_biggest_until(data_example_nan, 500.0)


    def test_setup_HDC_deltas_single(self):
        """
        Tests if contour is created with a single float for deltas
        as the exception should handle.
        """

        try:
            self._setup(deltas=0.05)

        except:
            print("contour couldn't be calculated")


    def test_setup_HDC_deltas_none(self):
        """
        Tests if default deltas are correctly calcualted.
        """

        test_contour_HDC = self._setup(deltas=None)
        expected =  [0.01] * test_contour_HDC.distribution.n_dim
        np.testing.assert_allclose(test_contour_HDC.deltas, expected, atol=0.001)


    def test_setup_HDC_deltas_value(self):
        """
        Tests error when length of deltas is not equal with number of dimensions.
        """

        with self.assertRaises(ValueError):
            self._setup(deltas=[0.05, 0.05, 0.05])


    def test_setup_HDC_limits_length(self):
        """
        Tests error when length of limits is not equal with number of dimensions.
        """

        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20), (0, 20), (0, 20)])


    def test_setup_HDC_limits_none(self):
        """
        Tests error when length of limits is not equal with number of dimensions.
        """
        test_contour_HDC = self._setup(limits=None)
        expected =  [(0, 19.18)] * test_contour_HDC.distribution.n_dim
        np.testing.assert_allclose(test_contour_HDC.limits, expected, atol=0.1)


    def test_setup_HDC_limits_Tuple_length(self):
        """
        Tests error when length of limits_tuples is not two.
        """

        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20), (20)])


        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20, 1), (0, 20)])


# This is commented out as it does not ensure that the sorting algorithm
# functions as intended.
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


class DirectSamplingTest(unittest.TestCase):
    def _setup(self,
               dep1=(None, None, None),
               dep2=(0, None, 0),
               par1=(ConstantParam(1.471), ConstantParam(0.8888),
                     ConstantParam(2.776)),
               par2=(FunctionParam('exp3', 0.0400, 0.1748, -0.2243), None,
                     FunctionParam('power3', 0.1, 1.489, 0.1901))
               ):
        """
        Create a 1-year contour (same as in DOI: 10.1016/j.oceaneng.2012.12.034).
        """
        return_period = 1 # years
        state_duration = 6 # hours

        # Define dependency tuple.
        self.dep1 = dep1
        self.dep2 = dep2

        # Define parameters.
        self.par1 = par1
        self.par2 = par2

        # Create distributions.
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(sigma=par2[0], mu=par2[2])

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Calculate contour.
        ds_contour = DirectSamplingContour(
            mul_dist, return_period, state_duration, 500000, 6)
        return ds_contour

    def test_direct_sampling_contour(self):
        """
        Computes a direct sampling contour and compares it with results
        presented in the literature (DOI: 10.1016/j.oceaneng.2012.12.034).
        """

        # Fix the random seed for consistency in repeated tests.
        prng = np.random.RandomState(42)

        contour = self._setup()
        ref_contour_hs_1 = \
            [9.99, 10.65, 10.99, 11.25, 11.25, 11.41, 11.42, 11.46, 11.48,
             11.54, 11.57, 11.56, 11.58, 11.59, 11.59, 11.60, 11.60, 11.59,
             11.59, 11.56, 11.53, 11.46, 11.26, 10.88, 7.44, 2.05]
        ref_contour_tz_1 = \
            [12.34, 12.34, 12.31, 12.25, 12.25, 12.18, 12.17, 12.15, 12.13,
             12.06, 12.02, 12.03, 12.00, 11.96, 11.95, 11.86, 11.84, 11.77,
             11.76, 11.67, 11.60, 11.47, 11.20, 10.77, 7.68, 3.76]
        np.testing.assert_allclose(contour.coordinates[0][0:26], ref_contour_hs_1, atol=0.5)
        np.testing.assert_allclose(contour.coordinates[1][0:26], ref_contour_tz_1, atol=0.5)

    def test_3d_ds_contour(self):
        """
        Tests whether calculating a 3D DS contour raises the right error.
        """
        par = (ConstantParam(1.471), ConstantParam(0.8888),
                ConstantParam(2.776))
        dist = WeibullDistribution(*par)
        dep = (None, None, None)
        distributions = (dist, dist, dist)
        dependencies = (dep, dep, dep)
        mul_dist = MultivariateDistribution(distributions, dependencies)

        # Calculate a 3D contour, which should raise an error.
        self.assertRaises(NotImplementedError,
                          DirectSamplingContour, mul_dist, 1, 3, 500000, 6)

if __name__ == '__main__':
    unittest.main()
