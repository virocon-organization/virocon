import unittest
import csv
import numpy as np

from viroconcom.fitting import Fit


def read_benchmark_dataset(path='tests/testfiles/1year_dataset_A.txt'):
    """
    Reads a datasets provided for the environmental contour benchmark.
    Parameters
    ----------
    path : string
        Path to dataset including the file name, defaults to 'examples/datasets/A.txt'
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.
    x_label : str
        Label of the environmantal variable 1.
    y_label : str
        Label of the environmental variable 2.
    """

    x = list()
    y = list()
    x_label = None
    y_label = None
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx == 0:
                x_label = row[1][
                          1:]  # Ignore first char (is a white space).
                y_label = row[2][
                          1:]  # Ignore first char (is a white space).
            if idx > 0:  # Ignore the header
                x.append(float(row[1]))
                y.append(float(row[2]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y, x_label, y_label)


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
        dist_description_0 = {'name': 'Weibull_3p',
                              'dependency': (None, None, None),
                              'width_of_intervals': 2}
        dist_description_1 = {'name': 'Lognormal',
                              'dependency': (None, None, 0),
                              'functions': (None, None, 'exp3')}

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

        # Now use a 2-parameter Weibull distribution instead of 3-p distr.
        dist_description_0 = {'name': 'Weibull_2p',
                              'dependency': (None, None, None),
                              'width_of_intervals': 2}
        dist_description_1 = {'name': 'Lognormal',
                              'dependency': (None, None, 0),
                              'functions': (None, None, 'exp3')}
        my_fit = Fit((sample_1, sample_2),
                     (dist_description_0, dist_description_1))


    def test_2d_exponentiated_wbl_fit(self):
        """
        Tests if a 2D fit that includes an exp. Weibull distribution works.
        """
        prng = np.random.RandomState(42)

        # Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
        # which represents significant wave height.
        sample_hs = prng.weibull(1.5, 1000)*3

        # Let the second sample, which represents zero-upcrossing period increase
        # with significant wave height and follow a Lognormal distribution with
        # mean=2 and sigma=0.2
        sample_tz = [0.1 + 1.5 * np.exp(0.2 * point) +
                    prng.lognormal(2, 0.2) for point in sample_hs]


        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_hs = {'name': 'Weibull_Exp',
                               'dependency': (None, None, None, None),
                               # Shape, Location, Scale, Shape2
                               'width_of_intervals': 0.5}
        dist_description_tz = {'name': 'Lognormal_SigmaMu',
                               'dependency': (0, None, 0),
                               # Shape, Location, Scale
                               'functions': ('exp3', None, 'power3')
                               # Shape, Location, Scale
                               }

        # Fit the model to the data, first test a 1D fit.
        fit = Fit(sample_hs, dist_description_hs)
        # Now perform the 2D fit.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))

        dist0 = fit.mul_var_dist.distributions[0]
        dist1 = fit.mul_var_dist.distributions[1]

        self.assertGreater(dist0.shape(0), 1) # Should be about 1.5.
        self.assertLess(dist0.shape(0), 2)
        self.assertIsNone(dist0.loc(0)) # Has no location parameter, should be None.
        self.assertGreater(dist0.scale(0), 2) # Should be about 3.
        self.assertLess(dist0.scale(0), 4)
        self.assertGreater(dist0.shape2(0), 0.5) # Should be about 1.
        self.assertLess(dist0.shape2(0), 2)


    def test_fit_lnsquare2(self):
        """
        Tests a 2D fit that includes an logarithm square dependence function.
        """

        sample_hs, sample_tz, label_hs, label_tz = read_benchmark_dataset()


        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_hs = {'name': 'Weibull_Exp',
                               'dependency': (None, None, None, None),
                               # Shape, Location, Scale, Shape2
                               'width_of_intervals': 0.5}
        dist_description_tz = {'name': 'Lognormal_SigmaMu',
                               'dependency': (0, None, 0),
                               # Shape, Location, Scale
                               'functions': ('exp3', None, 'lnsquare2')
                               # Shape, Location, Scale
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))


        # Check whether the logarithmic square fit worked correctly.
        dist1 = fit.mul_var_dist.distributions[1]
        self.assertGreater(dist1.scale.a, 1) # Should be about 1-5
        self.assertLess(dist1.scale.a, 5)  # Should be about 1-5
        self.assertGreater(dist1.scale.b, 2) # Should be about 2-10
        self.assertLess(dist1.scale.b, 10)  # Should be about 2-10
        self.assertGreater(dist1.scale(0), 0.1)
        self.assertLess(dist1.scale(0), 10)
        self.assertEqual(dist1.scale.func_name, 'lnsquare2')


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
        dist_description_1 = {'name': 'Lognormal',
                              'dependency': (None, None, 0),
                              'functions': (None, None, 'exp3')}

        # Compute the fit.
        my_fit = Fit((sample_1, sample_2),
                     (dist_description_0, dist_description_1),
                     timeout=10)
