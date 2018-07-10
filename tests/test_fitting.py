import unittest

import numpy as np

from viroconcom.fitting import Fit


class FittingTest(unittest.TestCase):

    def test_2d_fit(self):
        """
        2-d Fit with Weibull and Lognormal distribution.
        """
        prng = np.random.RandomState(42)

        # Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
        # which represents significant wave height.
        sample_1 = prng.weibull(1.5, 1000)*3

        # Let the second sample, which represents spectral peak period increase
        # with significant wave height and follow a Lognormal distribution with
        # mean=2 and sigma=0.2
        sample_2 = [0.1 + 1.5 * np.exp(0.2 * point) +
                    prng.lognormal(2, 0.2) for point in sample_1]


        # Describe the distribution that should be fitted to the sample.
        dist_description_0 = {'name': 'Weibull',
                              'dependency': (None, None, None),
                              'width_of_intervals': 2}
        dist_description_1 = {'name': 'Lognormal_1',
                              'dependency': (None, None, 0),
                              'functions': (None, None, 'f2')}

        # Compute the fit.
        my_fit = Fit((sample_1, sample_2),
                     (dist_description_0, dist_description_1))
        dist0 = my_fit.mul_var_dist.distributions[0]
        dist1 = my_fit.mul_var_dist.distributions[1]
        self.assertAlmostEqual(dist0.shape(0), 1.4165147571863412, places=5)
        self.assertAlmostEqual(dist0.scale(0), 2.833833521811032, places=5)
        self.assertAlmostEqual(dist0.loc(0), 0.07055663251419833, places=5)
        self.assertAlmostEqual(dist1.shape(0), 0.17742685807554776 , places=5)
        #self.assertAlmostEqual(dist1.scale, 7.1536437634240135+2.075539206642004e^{0.1515051024957754x}, places=5)
        self.assertAlmostEqual(dist1.loc, None, places=5)

    def test_multi_processing(selfs):
        """
        2-d Fit with multiprocessing (specified by setting a value for timeout)
        """

        # Define a sample and a fit.
        prng = np.random.RandomState(42)
        sample_1 = prng.weibull(1.5, 1000)*3
        sample_2 = [0.1 + 1.5 * np.exp(0.2 * point) +
                    prng.lognormal(2, 0.2) for point in sample_1]
        dist_description_0 = {'name': 'Weibull',
                              'dependency': (None, None, None),
                              'width_of_intervals': 2}
        dist_description_1 = {'name': 'Lognormal_1',
                              'dependency': (None, None, 0),
                              'functions': (None, None, 'f2')}

        # Compute the fit.
        my_fit = Fit((sample_1, sample_2),
                     (dist_description_0, dist_description_1),
                     timeout=10)
