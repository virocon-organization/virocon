import unittest


import numpy as np
import scipy.stats as sts

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (
    WeibullDistribution, ExponentiatedWeibullDistribution,
    LognormalDistribution, NormalDistribution,
    MultivariateDistribution)
from viroconcom.fitting import Fit

class MultivariateDistributionTest(unittest.TestCase):
    """
    Create an example MultivariateDistribution (Vanem2012 model).
    """

    # Create a MultivariateDistribution, the joint distribution for Hs-Tz
    # presented in Vanem et al. (2012).
    # Start with a Weibull distribution for wave height, Hs.
    shape = ConstantParam(1.471)
    loc = ConstantParam(0.8888)
    scale = ConstantParam(2.776)
    par0 = (shape, loc, scale)
    dist0 = WeibullDistribution(*par0)

    # Conditional lognormal distribution for period, Tz.
    mu = FunctionParam('power3', 0.1, 1.489, 0.1901)
    sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)
    dist1 = LognormalDistribution(mu=mu, sigma=sigma)

    distributions = [dist0, dist1]

    dep0 = (None, None, None)
    dep1 = (0, None, 0)
    dependencies = [dep0, dep1]

    mul_var_dist = MultivariateDistribution(distributions, dependencies)


    def test_add_distribution_err_msg(self):
        """
        Tests if the right exception is raised when distribution1 has a
        dependency.
        """

        dep0 = (None, 0, None)
        dependencies = [dep0, self.dep1]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)


    def test_add_distribution_iter(self):
        """
        Tests if an exception is raised by the function add_distribution when
        distributions isn't iterable but dependencies is and the other way around.
        """

        distributions = 1
        with self.assertRaises(ValueError):
            MultivariateDistribution(distributions, self.dependencies)
        dependencies = 0
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)

    def test_add_distribution_length(self):
        """
        Tests if an exception is raised when distributions and dependencies
        are of unequal length.
        """

        dep2 = (0, None, None)
        dependencies = [self.dep0, self.dep1, dep2]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)

    def test_add_distribution_dependencies_length(self):
        """
        Tests if an exception is raised when a tuple in dependencies
        has not length 3.
        """

        dep0 = (None, None)
        dependencies = [dep0, self.dep1]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)

    def test_add_distribution_dependencies_value(self):
        """
        Tests if an exception is raised when dependencies has an invalid value.
        """

        dep0 = (-3, None, None)
        dependencies = [dep0, self.dep1]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)

    def test_add_distribution_not_iterable(self):
        """
        Tests the function when both distributions and dependencies
        are not iterable.
        """

        distributions = 1
        dependencies = 2
        with self.assertRaises(ValueError):
            MultivariateDistribution(distributions, dependencies)

    def test_multivariate_draw_sample(self):
        """
        Tests the draw_sample() function of MulvariateDistribution.
        """
        n=10000
        sample = self.mul_var_dist.draw_sample(n)
        self.assertEqual(n, sample[0].size)
        self.assertEqual(n, sample[1].size)

        # Fit the sample to the correct model structure and compare the
        # estimated parameters with the true parameters.
        dist_description_0 = {'name': 'Weibull',
                              'dependency': (None, None, None),
                              'width_of_intervals': 0.5}
        dist_description_1 = {'name': 'Lognormal',
                              'dependency': (0, None, 0),
                              'functions': ('exp3', None, 'power3')}
        fit = Fit(sample, [dist_description_0, dist_description_1])
        fitted_dist0 = fit.mul_var_dist.distributions[0]
        fitted_dist1 = fit.mul_var_dist.distributions[1]
        self.assertAlmostEqual(fitted_dist0.shape(0), self.shape(0), delta=0.1)
        self.assertAlmostEqual(fitted_dist0.loc(0), self.loc(0), delta=0.1)
        self.assertAlmostEqual(fitted_dist0.scale(0), self.scale(0), delta=0.1)
        self.assertAlmostEqual(fitted_dist1.shape.a, 0.04, delta=0.1)
        self.assertAlmostEqual(fitted_dist1.shape.b, 0.1748, delta=0.1)
        self.assertAlmostEqual(fitted_dist1.shape.c, -0.2243, delta=0.15)

    def test_multivariate_cdf(self):
        """
         Tests the cdf() function of MulvariateDistribution.
         """
        p = self.mul_var_dist.cdf([2, 2])
        self.assertAlmostEqual(p, 0, delta=0.01)

        p = self.mul_var_dist.cdf([[2, 20], [2, 16]])
        np.testing.assert_allclose(p, [0, 1], atol=0.01)

    def test_multivariate_pdf(self):
        """
        Tests the pdf() function of MulvariateDistribution.
        """
        # Let's compare the density values with density values presented in
        # Haselsteiner et al. (2017), Fig. 5, DOI: 10.1016/j.coastaleng.2017.03.002
        f = self.mul_var_dist.pdf([10, 13])
        self.assertAlmostEqual(f, 0.000044, delta=0.00002)

        f = self.mul_var_dist.pdf([[8, 10, 12], [12.5, 13, 13.4]])
        np.testing.assert_allclose(f, [0.000044, 0.000044, 0.000044], atol=0.00002)

    def test_latex_representation(self):
        """
        Tests if the latex representation is correct.
        """
        m = self.mul_var_dist
        computed_latex = m.latex_repr(['Hs', 'Tp'])
        correct_latex = \
        ['\\text{ joint PDF: }',
         'f(h_{s},t_{p})=f_{H_{s}}(h_{s})f_{T_{p}|H_{s}}(t_{p}|h_{s})', '',
         '1\\text{. variable, }H_{s}: ',
         'f_{H_{s}}(h_{s})=\\dfrac{\\beta_{h_{s}}}{\\alpha_{h_{s}}}'
         '\\left(\\dfrac{h_{s}-\\gamma_{h_{s}}}{\\alpha_{h_{s}}}\\right)^'
         '{\\beta_{h_{s}}-1}\\exp\\left[-\\left(\\dfrac{h_{s}-'
         '\\gamma_{h_{s}}}{\\alpha_{h_{s}}}\\right)^{\\beta_{h_{s}}}'
         '\\right]', '\\quad\\text{ with }\\alpha_{h_{s}}=2.776,',
         '\\quad\\qquad\\;\\; \\beta_{h_{s}}=1.471,',
         '\\quad\\qquad\\;\\; \\gamma_{h_{s}}=0.8888.', '',
         '2\\text{. variable, }T_{p}: ',
         'f_{T_{p}|H_{s}}(t_{p}|h_{s})=\\dfrac{1}'
         '{t_{p}\\tilde{\\sigma}_{t_{p}}\\sqrt{2\\pi}}\\exp\\left[-'
         '\\dfrac{(\\ln t_{p}-\\tilde{\\mu}_{t_{p}})^2}'
         '{2\\tilde{\\sigma}_{t_{p}}^2}\\right]',
         '\\quad\\text{ with }\\tilde{\\mu}_{t_{p}}=0.1 + 1.489h_{s}^(0.1901),',
         '\\quad\\qquad\\;\\; \\tilde{\\sigma}_{t_{p}}=0.04 + 0.1748e^{-0.2243h_{s}}.']

        self.assertEqual(computed_latex, correct_latex)


class ParametricDistributionTest(unittest.TestCase):

    def test_distribution_shape_None(self):
        """
        Tests if shape is set to default when it has value 'None'.
        """

        # Define parameters.
        shape = None
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)

        dist = NormalDistribution(*par1)
        shape_test = dist._get_parameter_values(rv_values, dependencies)[0]
        self.assertEqual(shape_test, 1)


    def test_distribution_loc_None(self):
        """
        Tests if loc is set to default when it has value 'None'.
        """

        # Define parameters.
        shape = ConstantParam(0.8888)
        loc = None
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)

        dist = WeibullDistribution(*par1)
        loc_test = dist._get_parameter_values(rv_values, dependencies)[1]
        self.assertEqual(loc_test, 0)


    def test_distribution_loc_scale(self):
        """
        Tests if scale is set to default when it has value 'None'.
        """

        # Define parameters.
        shape = ConstantParam(0.8888)
        loc = ConstantParam(2.776)
        scale = None
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)

        dist = NormalDistribution(*par1)
        scale_test = dist._get_parameter_values(rv_values, dependencies)[2]
        self.assertEqual(scale_test, 1)


    def test_check_parameter_value(self):
        """
        Tests if the right exception is raised when the given parameters are
        not in the valid range of numbers.
        """

        shape = None
        loc = ConstantParam(0.8888)
        scale = ConstantParam(-2.776)
        par1 = (shape, loc, scale)

        dist = WeibullDistribution(*par1)

        with self.assertRaises(ValueError):
            dist._check_parameter_value(2, -2.776)
        with self.assertRaises(ValueError):
            dist._check_parameter_value(2, np.inf)

    def test_parameter_name_to_index(self):
        dist = ExponentiatedWeibullDistribution()
        self.assertEqual(dist.param_name_to_index('shape'), 0)
        self.assertEqual(dist.param_name_to_index('loc'), 1)
        self.assertEqual(dist.param_name_to_index('scale'), 2)
        self.assertEqual(dist.param_name_to_index('shape2'), 3)
        self.assertRaises(ValueError, dist.param_name_to_index, 'something')

    def test_exponentiated_weibull_distribution_cdf(self):
        """
        Tests the CDF of the exponentiated Weibull distribution.
        """

        # Define parameters with the values from the distribution fitted to
        # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
        shape = ConstantParam(0.4743)
        loc = None
        scale = ConstantParam(0.0373)
        shape2 = ConstantParam(46.6078)
        params = (shape, loc, scale, shape2)
        dist = ExponentiatedWeibullDistribution(*params)

        # CDF(1) should be roughly 0.7, see Figure 12 in
        # https://arxiv.org/pdf/1911.12835.pdf .
        p = dist.cdf(1)
        self.assertGreater(p, 0.5)
        self.assertLess(p, 0.8)

        # CDF(4) should be roughly 0.993, see Figure 12 in
        # https://arxiv.org/pdf/1911.12835.pdf .
        ps = dist.cdf(np.array((0.5, 1, 2, 4)))
        self.assertGreater(ps[-1], 0.99)
        self.assertLess(ps[-1], 0.999)

        p = dist.cdf(-1)
        self.assertEqual(p, 0) # CDF(negative value) should be 0


    def test_exponentiated_weibull_distribution_icdf(self):
        """
        Tests the ICDF of the exponentiated Weibull distribution.
        """

        # Define parameters with the values from the distribution fitted to
        # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
        shape = ConstantParam(0.4743)
        loc = None
        scale = ConstantParam(0.0373)
        shape2 = ConstantParam(46.6078)
        params = (shape, loc, scale, shape2)
        dist = ExponentiatedWeibullDistribution(*params)

        # ICDF(0.5) should be roughly 0.8, see Figure 12 in
        # https://arxiv.org/pdf/1911.12835.pdf .
        x = dist.i_cdf(0.5)
        self.assertGreater(x, 0.5)
        self.assertLess(x, 1)
        x = dist.ppf(0.5)
        self.assertGreater(x, 0.5)
        self.assertLess(x, 1)

        # ICDF(0.9) should be roughly 1.8, see Figure 12
        # in https://arxiv.org/pdf/1911.12835.pdf .
        xs = dist.i_cdf(np.array((0.1, 0.2, 0.5, 0.9)))
        self.assertGreater(xs[-1], 1)
        self.assertLess(xs[-1], 2)

        p = dist.i_cdf(5)
        self.assertTrue(np.isnan(p)) # ICDF(value greater than 1) should be nan.


    def test_exponentiated_weibull_distribution_pdf(self):
        """
        Tests the PDF of the exponentiated Weibull distribution.
        """

        # Define parameters with the values from the distribution fitted to
        # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
        shape = ConstantParam(0.4743)
        loc = None
        scale = ConstantParam(0.0373)
        shape2 = ConstantParam(46.6078)
        params = (shape, loc, scale, shape2)
        dist = ExponentiatedWeibullDistribution(*params)

        # PDF(0.7) should be roughly 1.1, see Figure 3 in
        # https://arxiv.org/pdf/1911.12835.pdf .
        x = dist.pdf(0.7)
        self.assertGreater(x, 0.6)
        self.assertLess(x, 1.5)

        # PDF(2) should be roughly 0.1, see Figure 12
        # in https://arxiv.org/pdf/1911.12835.pdf .
        xs = dist.pdf(np.array((0.1, 0.5, 2)))
        self.assertGreater(xs[-1], 0.05)
        self.assertLess(xs[-1], 0.2)

        p = dist.pdf(-1)
        self.assertEqual(p, 0) # PDF(value less than 0) should be 0.

    def test_fitting_exponentiated_weibull(self):
        """
        Tests estimating the parameters of the  exponentiated Weibull distribution.
        """

        dist = ExponentiatedWeibullDistribution()

        # Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
        # which represents significant wave height.
        hs = sts.weibull_min.rvs(1.5, loc=0, scale=3, size=1000, random_state=42)

        params = dist.fit(hs)
        self.assertGreater(params[0], 1) # shape parameter should be about 1.5.
        self.assertLess(params[0], 2)
        self.assertIsNone(params[1], 2) # location parameter should be None.
        self.assertGreater(params[2], 2) # scale parameter should be about 3.
        self.assertLess(params[2], 4)
        self.assertGreater(params[3], 0.5) # shape2 parameter should be about 1.
        self.assertLess(params[3], 2)


        params = dist.fit(hs, shape2=1)
        self.assertGreater(params[0], 1) # shape parameter should be about 1.5.
        self.assertLess(params[0], 2)
        self.assertIsNone(params[1], 2) # location parameter should be None.
        self.assertGreater(params[2], 2) # scale parameter should be about 3.
        self.assertLess(params[2], 4)

        # Check whether the fitted distribution has a working CDF and PDF.
        self.assertGreater(dist.cdf(2), 0)
        self.assertGreater(dist.pdf(2), 0)

    def test_fit_exponentiated_weibull_with_zero(self):
        """
        Tests fitting the exponentiated Weibull distribution if the dataset
        contains 0s.
        """

        dist = ExponentiatedWeibullDistribution()

        # Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
        # which represents significant wave height.
        hs = sts.weibull_min.rvs(1.5, loc=0, scale=3, size=1000, random_state=42)

        # Add zero-elements to the dataset.
        hs = np.append(hs, [0, 0, 1.3])

        params = dist.fit(hs)

        self.assertAlmostEquals(params[0], 1.5, delta=0.5)
        self.assertIsNone(params[1], 2) # location parameter should be None.
        self.assertAlmostEquals(params[2], 3, delta=1)
        self.assertAlmostEquals(params[3], 1, delta=0.5)


if __name__ == '__main__':
    unittest.main()
