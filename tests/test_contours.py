#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:49:33 2017

@author: nb
"""

import unittest
import os

import numpy as np
import pandas as pd

from .context import viroconcom

from viroconcom.params import ConstantParam, FunctionParam

from viroconcom.distributions import (WeibullDistribution, LognormalDistribution,
                                    NormalDistribution, MultivariateDistribution)
from viroconcom.contours import IFormContour, HighestDensityContour


_here = os.path.dirname(__file__)
testfiles_path = os.path.abspath(os.path.join(_here, "testfiles"))

class HDCCreationTest(unittest.TestCase):


    def test_HDC2d_WL(self):
        """
        Creating Contour example for 2-d HDC with Weibull and Lognormal
        distribution
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam(0.1000, 1.489, 0.1901, 'f1')
        sigma = FunctionParam(0.0400, 0.1748, -0.2243, 'f2')

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

        finaldt0 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0][0],
                                'y' : test_contour_HDC.coordinates[0][1]})


        result0 = pd.read_csv(testfiles_path + "/HDC2dWL_coordinates.csv")

        for g,h in [(g, h) for g in result0.index for h in result0.columns]:
            self.assertAlmostEqual(result0.ix[g, h], finaldt0.ix[g, h], places=8)



    def test_HDC3d_WLL(self):
        """
        Creating Contour example for 3-d HDC with Weibull, Lognormal and
        Lognormal distribution
        """

        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam(0.1000, 1.489, 0.1901, "f1")
        sigma = FunctionParam(0.0400, 0.1748, -0.2243, "f2")


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

        finaldt = pd.DataFrame({'x' : test_contour_HDC.coordinates[0][0],
                                'y' : test_contour_HDC.coordinates[0][1],
                                'z' : test_contour_HDC.coordinates[0][2]})


        result = pd.read_csv(testfiles_path + "/HDC3dWLL_coordinates.csv")
        for i,j in [(i, j) for i in result.index for j in result.columns]:
            self.assertAlmostEqual(result.ix[i,j], finaldt.ix[i,j], places=8)


    def test_HDC4d_WLLL(self):
        """
        Creating Contour example for 4-d HDC with Weibull, Lognormal,
        Lognormal and Lognormal distribution
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)
        dep4 = (0, None, 0)

        #define parameters
        shape = ConstantParam(2.776)
        loc = ConstantParam(1.471)
        scale = ConstantParam(0.8888)
        par1 = (shape, loc, scale)

        mu = FunctionParam(0.1000, 1.489, 0.1901, "f1")
        sigma = FunctionParam(0.0400, 0.1748, -0.2243, "f2")

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)
        dist3 = LognormalDistribution(mu=mu, sigma=sigma)
        dist4 = LognormalDistribution(mu=mu, sigma=sigma)


        distributions = [dist1, dist2, dist3, dist4]
        dependencies = [dep1, dep2, dep3, dep4]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        #del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
        #calc contour
        n_years = 50
        limits = [(0, 20), (0, 18), (0, 18), (0, 18)]
        deltas = [1, 1, 1, 1]

        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)



    def test_HDC2d_WN(self):
        """
        Creating Contour example
        """


        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (None, 0, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam(4, 10, 0.02, "f1")
        scale = FunctionParam(0.1, 0.02, -0.1, "f2")
        par2 = (shape, loc, scale)

        #del shape, loc, scale

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = NormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        #del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
        #calc contour
        n_years = 50
        limits = [(0, 20), (0, 20)]
        deltas = [0.05, 0.01]
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)

        finaldt2 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0][0],
                                 'y' : test_contour_HDC.coordinates[0][1]})

        result2 = pd.read_csv(testfiles_path + "/HDC2dWN_coordinates.csv")

        for k,l in [(k, l) for k in result2.index for l in result2.columns]:
            self.assertAlmostEqual(result2.ix[k,l], finaldt2.ix[k,l], places=8)



    def test_HDC3d_WLN(self):

        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (None, 0, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam(4, 10, 0.02, "f1")
        scale = FunctionParam(0.1, 0.02, -0.1, "f2")
        par2 = (shape, loc, scale)

        mu = FunctionParam(0.1, 1.5, 0.2, "f1")
        sigma = FunctionParam(0.1, 0.2, -0.2, "f2")

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

        finaldt3 = pd.DataFrame({'x' : test_contour_HDC.coordinates[0][0],
                                 'y' : test_contour_HDC.coordinates[0][1],
                                 'z' : test_contour_HDC.coordinates[0][2]})

        matlab3 = pd.read_csv(testfiles_path + "/hdc3d_wln.csv", names=['x', 'y', 'z'])

        result3 = pd.read_csv(testfiles_path + "/HDC3dWLN_coordinates.csv")
        for m,n in [(m, n) for m in result3.index for n in result3.columns]:
            self.assertAlmostEqual(result3.ix[m, n], finaldt3.ix[m, n], places=8)



    def test_IForm2d_WL(self):
        """
        Creating Contour example
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam(0.1000, 1.489, 0.1901, "f1")
        sigma = FunctionParam(0.0400, 0.1748, -0.2243, "f2")

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_IForm = IFormContour(mul_dist, 50, 3, 400)

        finaldt4 = pd.DataFrame({'x' : test_contour_IForm.coordinates[0][0],
                                 'y' : test_contour_IForm.coordinates[0][1]})

        result4 = pd.read_csv(testfiles_path + "/IForm2dWL_coordinates.csv")
        for o,p in [(o, p) for o in result4.index for p in result4.columns]:
            self.assertAlmostEqual(result4.ix[o, p], finaldt4.ix[o, p], places=8)




    def test_IForm2d_WN(self):
        """
        Creating Contour example
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (None, 0, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        shape = None
        loc = FunctionParam(7, 1.489, 0.1901, "f1")
        scale = FunctionParam(1.5, 0.1748, -0.2243, "f2")
        par2 = (shape, loc, scale)

        #del shape, loc, scale

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = NormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_IForm = IFormContour(mul_dist, 50, 3, 400)

        finaldt5 = pd.DataFrame({'x' : test_contour_IForm.coordinates[0][0],
                                 'y' : test_contour_IForm.coordinates[0][1]})

        result5 = pd.read_csv(testfiles_path + "/IForm2dWN_coordinates.csv")

        for r,s in [(r, s) for r in result5.index for s in result5.columns]:
          self.assertAlmostEqual(result5.ix[r, s], finaldt5.ix[r, s], places=8)



    def test_IForm3d(self): # TODO what does this test do
        """
        Creating Contour example
        """

        #define dependency tuple
        dep1 = (None, None, None)
        dep2 = (0, None, 0)
        dep3 = (0, None, 0)

        #define parameters
        shape = ConstantParam(1.471)
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)

        mu = FunctionParam(0.1000, 1.489, 0.1901, "f1")
        sigma = FunctionParam(0.0400, 0.1748, -0.2243, "f2")

        #del shape, loc, scale

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(mu=mu, sigma=sigma)
        dist3 = LognormalDistribution(mu=mu, sigma=sigma)
        distributions = [dist1, dist2, dist3]
        dependencies = [dep1, dep2, dep3]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        test_contour_IForm = IFormContour(mul_dist, 50, 3, 400)



class HDCTest(unittest.TestCase):

    def _setup(self, limits=[(0, 20), (0, 20)], deltas=[0.05, 0.05],
                n_years = 25, dep1=(None, None, None), dep2=(0, None, 0),
                par1=(ConstantParam(1.471), ConstantParam(0.8888),
                ConstantParam(2.776)),
                par2=(FunctionParam(0.0400, 0.1748, -0.2243, "f2"), None,
                FunctionParam(0.1, 1.489, 0.1901, "f1"))):
        """
        Creating Contour example
        """

        self.limits = limits
        self.deltas = deltas
        self.n_years = n_years

        #define dependency tuple
        self.dep1 = dep1
        self.dep2 = dep2

        #define parameters
        self.par1 = par1
        self.par2 = par2

        #create distributions
        dist1 = WeibullDistribution(*par1)
        dist2 = LognormalDistribution(*par2)

        distributions = [dist1, dist2]
        dependencies = [dep1, dep2]

        mul_dist = MultivariateDistribution(distributions, dependencies)

        #calc contour
        test_contour_HDC = HighestDensityContour(mul_dist, n_years, 3,
                                                 limits, deltas)
        return test_contour_HDC


    def test_cumsum(self):
        """
        tests if the return values of cumsum_biggest_until are correct
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
        tests if ValueError is raised when the array has a 'nan' entry
        """

        test_contour_HDC = self._setup()
        data_example_nan = np.array([[80, 7, float('nan'), 40], [1, 9, 45, 23]])
        with self.assertRaises(ValueError):
            test_contour_HDC.cumsum_biggest_until(data_example_nan, 500.0)


    def test_setup_HDC_deltas_single(self):
        """
        tests if contour is created with a single float for deltas
        as the exception should handle
        """

        try:
            self._setup(deltas=0.05)

        except:
            print("contour couldn't be calculated")


    def test_setup_HDC_deltas_none(self):
        """
        tests error when length of deltas is not equal with number of dimensions
        """

        test_contour_HDC = self._setup(deltas=None)
        self.assertEqual(test_contour_HDC.deltas, [0.5] *
                         test_contour_HDC.distribution.n_dim)


    def test_setup_HDC_deltas_value(self):
        """
        tests error when length of deltas is not equal with number of dimensions
        """

        with self.assertRaises(ValueError):
            self._setup(deltas=[0.05, 0.05, 0.05])


    def test_setup_HDC_limits_length(self):
        """
        tests error when length of limits is not equal with number of dimensions
        """

        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20), (0, 20), (0, 20)])


    def test_setup_HDC_limits_none(self):
        """
        tests error when length of limits is not equal with number of dimensions
        """

        test_contour_HDC = self._setup(limits=None)
        self.assertEqual(test_contour_HDC.limits, [(0, 10)] *
                                    test_contour_HDC.distribution.n_dim)


    def test_setup_HDC_limits_Tuple_length(self):
        """
        tests error when length of limits_tuples is not two
        """

        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20), (20)])


        with self.assertRaises(ValueError):
            self._setup(limits=[(0, 20, 1), (0, 20)])


if __name__ == '__main__':
    unittest.main()