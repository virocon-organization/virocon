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

        self.assertEqual(str(my_fit)[0:5], 'Fit()')


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

    def test_fit_powerdecrease3(self):
        """
        Tests a 2D fit that includes an powerdecrease3 dependence function.
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
                               'functions': ('powerdecrease3', None, 'lnsquare2')
                               # Shape, Location, Scale
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))


        # Check whether the logarithmic square fit worked correctly.
        dist1 = fit.mul_var_dist.distributions[1]
        self.assertGreater(dist1.shape.a, -0.1) # Should be about 0
        self.assertLess(dist1.shape.a, 0.1)  # Should be about 0
        self.assertGreater(dist1.shape.b, 1.5) # Should be about 2-5
        self.assertLess(dist1.shape.b, 6)  # Should be about 2-10
        self.assertGreater(dist1.shape.c, 0.8) # Should be about 1.1
        self.assertLess(dist1.shape.c, 2)  # Should be about 1.1
        self.assertGreater(dist1.shape(0), 0.25) # Should be about 0.35
        self.assertLess(dist1.shape(0), 0.4) # Should be about 0.35
        self.assertEqual(dist1.shape.func_name, 'powerdecrease3')

    def test_fit_asymdecrease3(self):
        """
        Tests a 2D fit that includes an asymdecrease3 dependence function.
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
                               'functions': ('asymdecrease3', None, 'lnsquare2')
                               # Shape, Location, Scale
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))


        # Check whether the logarithmic square fit worked correctly.
        dist1 = fit.mul_var_dist.distributions[1]
        self.assertAlmostEqual(dist1.shape.a, 0, delta=0.1) # Should be about 0
        self.assertAlmostEqual(dist1.shape.b, 0.35, delta=0.4) # Should be about 0.35
        self.assertAlmostEqual(np.abs(dist1.shape.c), 0.45, delta=0.2) # Should be about 0.45
        self.assertAlmostEquals(dist1.shape(0), 0.35, delta=0.2) # Should be about 0.35

    def test_min_number_datapoints_for_fit(self):
        """
        Tests if the minimum number of datapoints required for a fit works.
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
                               'functions': ('exp3', None, 'lnsquare2'),
                               # Shape, Location, Scale
                               'min_datapoints_for_fit': 10
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))

        # Check whether the logarithmic square fit worked correctly.
        dist1 = fit.mul_var_dist.distributions[1]
        a_min_10 = dist1.scale.a

        # Now require more datapoints for a fit.
        dist_description_tz = {'name': 'Lognormal_SigmaMu',
                               'dependency': (0, None, 0),
                               # Shape, Location, Scale
                               'functions': ('exp3', None, 'lnsquare2'),
                               # Shape, Location, Scale
                               'min_datapoints_for_fit': 500
                               }

        # Fit the model to the data.
        fit = Fit((sample_hs, sample_tz),
                  (dist_description_hs, dist_description_tz))

        # Check whether the logarithmic square fit worked correctly.
        dist1 = fit.mul_var_dist.distributions[1]
        a_min_500 = dist1.scale.a

        # Because in case 2 fewer bins have been used we should get different
        # coefficients for the dependence function.
        self.assertNotEqual(a_min_10, a_min_500)

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

    def test_wbl_fit_with_negative_location(self):
        """
        Tests fitting a translated Weibull distribution which would result
        in a negative location parameter.
        """

        sample_hs, sample_tz, label_hs, label_tz = read_benchmark_dataset()


        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_hs = {'name': 'Weibull_3p',
                               'dependency': (None, None, None)}

        # Fit the model to the data.
        fit = Fit((sample_hs, ),
                  (dist_description_hs, ))


        # Correct values for 10 years of data can be found in
        # 10.1115/OMAE2019-96523 . Here we used 1 year of data.
        dist0 = fit.mul_var_dist.distributions[0]
        self.assertAlmostEqual(dist0.shape(0) / 10, 1.48 / 10, places=1)
        self.assertGreater(dist0.loc(0), 0.0) # Should be 0.0981
        self.assertLess(dist0.loc(0), 0.3)  # Should be 0.0981
        self.assertAlmostEqual(dist0.scale(0), 0.944, places=1)

        # Shift the wave data with -1 m and fit again.
        sample_hs = sample_hs - 2
        # Negative location values will be set to zero instead and a
        # warning will be raised.
        with self.assertWarns(RuntimeWarning):
            fit = Fit((sample_hs, ),
                      (dist_description_hs, ))
            dist0 = fit.mul_var_dist.distributions[0]
            self.assertAlmostEqual(dist0.shape(0) / 10, 1.48 / 10, places=1)

            # Should be estimated to be  0.0981 - 2 and corrected to be 0.
            self.assertEqual(dist0.loc(0), 0)

            self.assertAlmostEqual(dist0.scale(0), 0.944, places=1)

    def test_omae2020_wind_wave_model(self):
        """
        Tests fitting the wind-wave model that was used in the publication
        'Global hierarchical models for wind and wave contours' on dataset D.
        """

        sample_v, sample_hs, label_v, label_hs = read_benchmark_dataset(path='tests/testfiles/1year_dataset_D.txt')


        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'fixed_parameters' :  (None,         None, None,     5), # shape, location, scale, shape2
                              'dependency':        (0,            None, 0,        None), # shape, location, scale, shape2
                              'functions':         ('logistics4', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20}

        # Fit the model to the data.
        fit = Fit((sample_v, sample_hs),
                  (dist_description_v, dist_description_hs))



        dist0 = fit.mul_var_dist.distributions[0]
        self.assertAlmostEqual(dist0.shape(0), 2.42, delta=1)
        self.assertAlmostEqual(dist0.scale(0), 10.0, delta=2)
        self.assertAlmostEqual(dist0.shape2(0), 0.761, delta=0.5)

        dist1 = fit.mul_var_dist.distributions[1]
        self.assertEqual(dist1.shape2(0), 5)
        inspection_data1 = fit.multiple_fit_inspection_data[1]
        self.assertEqual(inspection_data1.shape2_value[0], 5)
        self.assertAlmostEqual(inspection_data1.shape_value[0], 0.8, delta=0.5) # interval centered at 1
        self.assertAlmostEqual(inspection_data1.shape_value[4], 1.5, delta=0.5)  # interval centered at 9
        self.assertAlmostEqual(inspection_data1.shape_value[9], 2.5, delta=1)  # interval centered at 19
        self.assertAlmostEqual(dist1.shape(0), 0.8, delta=0.3)
        self.assertAlmostEqual(dist1.shape(10), 1.6, delta=0.5)
        self.assertAlmostEqual(dist1.shape(20), 2.3, delta=0.7)
        self.assertAlmostEqual(dist1.shape.a, 0.582, delta=0.5)
        self.assertAlmostEqual(dist1.shape.b, 1.90, delta=1)
        self.assertAlmostEqual(dist1.shape.c, 0.248, delta=0.5)
        self.assertAlmostEqual(dist1.shape.d, 8.49, delta=5)
        self.assertAlmostEqual(inspection_data1.scale_value[0], 0.15, delta=0.2) # interval centered at 1
        self.assertAlmostEqual(inspection_data1.scale_value[4], 1, delta=0.5)  # interval centered at 9
        self.assertAlmostEqual(inspection_data1.scale_value[9], 4, delta=1)  # interval centered at 19
        self.assertAlmostEqual(dist1.scale(0), 0.15, delta=0.5)
        self.assertAlmostEqual(dist1.scale(10), 1, delta=0.5)
        self.assertAlmostEqual(dist1.scale(20), 4, delta=1)
        self.assertAlmostEqual(dist1.scale.a, 0.394, delta=0.5)
        self.assertAlmostEqual(dist1.scale.b, 0.0178, delta=0.1)
        self.assertAlmostEqual(dist1.scale.c, 1.88, delta=0.8)

    def test_wrong_model(self):
        """
        Tests wheter errors are raised when incorrect fitting models are
        specified.
        """

        sample_v, sample_hs, label_v, label_hs = read_benchmark_dataset(path='tests/testfiles/1year_dataset_D.txt')


        # This structure is incorrect as there is not distribution called 'something'.
        dist_description_v = {'name': 'something',
                              'dependency': (None, None, None, None),
                              'fixed_parameters': (None, None, None, None), # shape, location, scale, shape2
                              'width_of_intervals': 2}
        with self.assertRaises(ValueError):
            # Fit the model to the data.
            fit = Fit((sample_v, ),
                      (dist_description_v, ))


        # This structure is incorrect as there is not dependence function called 'something'.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'dependency':        (0, None, 0,  None), # shape, location, scale, shape2
                              'functions':         ('something', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20}
        with self.assertRaises(ValueError):
            # Fit the model to the data.
            fit = Fit((sample_v, sample_hs),
                      (dist_description_v, dist_description_hs))


        # This structure is incorrect as there will be only 1 or 2 intervals
        # that fit 2000 datapoints.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'dependency':        (0, None, 0,  None), # shape, location, scale, shape2
                              'functions':         ('logistics4', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 2000}
        with self.assertRaises(RuntimeError):
            # Fit the model to the data.
            fit = Fit((sample_v, sample_hs),
                      (dist_description_v, dist_description_hs))



        # This structure is incorrect as alpha3 is only compatible with
        # logistics4 .
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'fixed_parameters' :  (None,         None, None,     5), # shape, location, scale, shape2
                              'dependency':        (0,            None, 0,        None), # shape, location, scale, shape2
                              'functions':         ('power3', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20}
        with self.assertRaises(TypeError):
            # Fit the model to the data.
            fit = Fit((sample_v, sample_hs),
                      (dist_description_v, dist_description_hs))


        # This structure is incorrect as only shape2 of an exponentiated Weibull
        # distribution can be fixed at the moment.
        dist_description_v = {'name': 'Lognormal',
                              'dependency': (None, None, None, None),
                              'fixed_parameters': (None, None, 5, None), # shape, location, scale, shape2
                              'width_of_intervals': 2}
        with self.assertRaises(NotImplementedError):
            # Fit the model to the data.
            fit = Fit((sample_v, ),
                      (dist_description_v, ))

        # This structure is incorrect as only shape2 of an exponentiated Weibull
        # distribution can be fixed at the moment.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'fixed_parameters' :  (None,        None, 5,        None), # shape, location, scale, shape2
                              'dependency':        (0,            None, 0,        None), # shape, location, scale, shape2
                              'functions':         ('logistics4', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20}
        with self.assertRaises(NotImplementedError):
            # Fit the model to the data.
            fit = Fit((sample_v, sample_hs),
                      (dist_description_v, dist_description_hs))


    def test_weighting_of_dependence_function(self):
        """
        Tests if using weights when the dependence function is fitted works
        correctly.
        """

        sample_v, sample_hs, label_v, label_hs = read_benchmark_dataset(path='tests/testfiles/1year_dataset_D.txt')


        # Define the structure of the probabilistic model that will be fitted to the
        # dataset.
        dist_description_v = {'name': 'Weibull_Exp',
                              'dependency': (None, None, None, None),
                              'width_of_intervals': 2}
        dist_description_hs = {'name': 'Weibull_Exp',
                              'fixed_parameters' :  (None,        None, None,     5), # shape, location, scale, shape2
                              'dependency':        (0,            None, 0,        None), # shape, location, scale, shape2
                              'functions':         ('logistics4', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20,
                              'do_use_weights_for_dependence_function': False}

        # Fit the model to the data.
        fit = Fit((sample_v, sample_hs),
                  (dist_description_v, dist_description_hs))
        dist1_no_weights = fit.mul_var_dist.distributions[1]


        # Now perform a fit with weights.
        dist_description_hs = {'name': 'Weibull_Exp',
                              'fixed_parameters' :  (None,        None, None,     5), # shape, location, scale, shape2
                              'dependency':        (0,            None, 0,        None), # shape, location, scale, shape2
                              'functions':         ('logistics4', None, 'alpha3', None), # shape, location, scale, shape2
                              'min_datapoints_for_fit': 20,
                              'do_use_weights_for_dependence_function': True}
        # Fit the model to the data.
        fit = Fit((sample_v, sample_hs),
                  (dist_description_v, dist_description_hs))
        dist1_with_weights = fit.mul_var_dist.distributions[1]

        # Make sure the two fitted dependnece functions are different.
        d = np.abs(dist1_with_weights.scale(0) - dist1_no_weights.scale(0)) / \
            np.abs(dist1_no_weights.scale(0))
        self.assertGreater(d, 0.01)

        # Make sure they are not too different.
        d = np.abs(dist1_with_weights.scale(20) - dist1_no_weights.scale(20)) / \
            np.abs(dist1_no_weights.scale(20))
        self.assertLess(d, 0.5)
