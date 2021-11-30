import pytest
import numpy as np
import scipy.stats as sts

from virocon import ExponentiatedWeibullDistribution
from virocon import GeneralizedGammaDistribution


def test_exponentiated_weibull_distribution_cdf():
    """
    Tests the CDF of the exponentiated Weibull distribution.
    """
    # Define dist with parameters from the distribution fitted to
    # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
    dist = ExponentiatedWeibullDistribution(alpha=0.0373, beta=0.4743, delta=46.6078)

    # CDF(1) should be roughly 0.7, see Figure 12 in
    # https://arxiv.org/pdf/1911.12835.pdf .
    p = dist.cdf(1)
    np.testing.assert_allclose(p, 0.7, atol=0.1)

    # CDF(4) should be roughly 0.993, see Figure 12 in
    # https://arxiv.org/pdf/1911.12835.pdf .
    p = dist.cdf(4)
    assert p > 0.99
    assert p < 0.999

    # CDF(negative value) should be 0
    p = dist.cdf(-1)
    assert p == 0


def test_exponentiated_weibull_distribution_icdf():
    """
    Tests the ICDF of the exponentiated Weibull distribution.
    """
    # Define dist with parameters from the distribution fitted to
    # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
    dist = ExponentiatedWeibullDistribution(alpha=0.0373, beta=0.4743, delta=46.6078)

    # ICDF(0.5) should be roughly 0.8, see Figure 12 in
    # https://arxiv.org/pdf/1911.12835.pdf .
    x = dist.icdf(0.5)
    assert x > 0.5
    assert x < 1

    # ICDF(0.9) should be roughly 1.8, see Figure 12
    # in https://arxiv.org/pdf/1911.12835.pdf .
    x = dist.icdf(0.9)
    assert x > 1
    assert x < 2

    # ICDF(value greater than 1) should be nan.
    x = dist.icdf(5)
    assert np.isnan(x)


def test_exponentiated_weibull_distribution_pdf():
    """
    Tests the PDF of the exponentiated Weibull distribution.
    """
    # Define dist with parameters from the distribution fitted to
    # dataset A with the MLE in https://arxiv.org/pdf/1911.12835.pdf .
    dist = ExponentiatedWeibullDistribution(alpha=0.0373, beta=0.4743, delta=46.6078)

    # PDF(0.7) should be roughly 1.1, see Figure 3 in
    # https://arxiv.org/pdf/1911.12835.pdf .
    x = dist.pdf(0.7)
    assert x > 0.6
    assert x < 1.5

    # PDF(2) should be roughly 0.1, see Figure 12
    # in https://arxiv.org/pdf/1911.12835.pdf .
    x = dist.pdf(2)
    assert x > 0.05
    assert x < 0.2

    # PDF(value less than 0) should be 0.
    x = dist.pdf(-1)
    assert x == 0


def test_fitting_exponentiated_weibull():
    """
    Tests estimating the parameters of the  exponentiated Weibull distribution.
    """

    dist = ExponentiatedWeibullDistribution()

    # Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
    # which represents significant wave height.
    hs = sts.weibull_min.rvs(1.5, loc=0, scale=3, size=1000, random_state=42)

    dist.fit(hs, method="wlsq", weights="quadratic")

    # shape parameter/ beta should be about 1.5.
    assert dist.parameters["beta"] > 1
    assert dist.parameters["beta"] < 2

    # scale parameter / alpha should be about 3.
    assert dist.parameters["alpha"] > 2
    assert dist.parameters["alpha"] < 4

    # shape2 parameter / delta should be about 1.
    assert dist.parameters["delta"] > 0.5
    assert dist.parameters["delta"] < 2

    dist = ExponentiatedWeibullDistribution(f_delta=1)

    dist.fit(hs, method="wlsq", weights="quadratic")

    # shape parameter/ beta should be about 1.5.
    assert dist.parameters["beta"] > 1
    assert dist.parameters["beta"] < 2

    # scale parameter / alpha should be about 3.
    assert dist.parameters["alpha"] > 2
    assert dist.parameters["alpha"] < 4

    # shape2 parameter / delta should be 1.
    assert dist.parameters["delta"] == 1

    # Check whether the fitted distribution has a working CDF and PDF.
    assert dist.cdf(2) > 0
    assert dist.pdf(2) > 0


def test_fit_exponentiated_weibull_with_zero():
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

    dist.fit(hs, method="wlsq", weights="quadratic")

    assert dist.parameters["beta"] == pytest.approx(1.5, abs=0.5)
    assert dist.parameters["alpha"] == pytest.approx(3, abs=1)
    assert dist.parameters["delta"] == pytest.approx(1, abs=0.5)


def test_generalized_gamma_reproduce_Ochi_CDF():
    """
    Test reproducing the fitting of Ochi et. al (1980) and compare it to
    virocons implementation of the generalized gamma distribution. The results
    should be the same.

    """

    # Define dist with parameters from the distribution of Fig. 4b in
    # M.K. Ochi, J.E. Wahlen, New approach for estimating the severes
    # sea state from statistical data , Coast. Eng. Chapter 38 (1992) 
    # pp. 512-525.
    
    dist = GeneralizedGammaDistribution(1.60, 0.98, 1.37)

    # CDF(0.5) should be roughly 0.21, see Fig. 4b
    # CDF(1) should be roughly 0.55, see Fig. 4b
    # CDF(1.5) should be roughly 0.70, see Fig. 4b
    # CDF(2) should be roughly 0.83, see Fig. 4b
    # CDF(4) should be roughly 0.98, see Fig. 4b
    # CDF(6) should be roughly 0.995, see Fig. 4b

    p1 = dist.cdf(0.5)
    p2 = dist.cdf(1)
    p3 = dist.cdf(1.5)
    p4 = dist.cdf(2)
    p5 = dist.cdf(4)
    p6 = dist.cdf(6)

    np.testing.assert_allclose(p1, 0.21, atol=0.05)
    np.testing.assert_allclose(p2, 0.55, atol=0.05)
    np.testing.assert_allclose(p3, 0.70, atol=0.05)
    np.testing.assert_allclose(p4, 0.83, atol=0.05)
    np.testing.assert_allclose(p5, 0.98, atol=0.005)
    np.testing.assert_allclose(p6, 0.995, atol=0.005)

    # CDF(negative value) should be 0
    p = dist.cdf(-1)
    assert p == 0


def test_generalized_gamma_compare_scipy_to_virocon_PDF():
    """
    Test which compares the PDF of a fitted scipy distribution and compare it 
    to the PDF of a fitted virocon distribution. The results should be the 
    same since virocon uses scipy.

    """

    x = np.linspace(2, 15, num=100)

    # Create sample
    true_m = 5
    true_c = 2.42
    true_loc = 0
    true_lambda_ = 0.5
    gamma_samples = sts.gengamma.rvs(
        a=true_m,
        c=true_c,
        loc=true_loc,
        scale=1 / true_lambda_,
        size=100,
        random_state=42,
    )

    # Fit distribution with virocon
    my_gamma = GeneralizedGammaDistribution()
    my_gamma.fit(gamma_samples, method="mle")
    my_pdf = my_gamma.pdf(x)

    # Fit distribution with scipy
    ref_gamma = sts.gengamma.fit(gamma_samples, floc=0)
    ref_pdf = sts.gengamma.pdf(
        x, ref_gamma[0], ref_gamma[1], ref_gamma[2], ref_gamma[3]
    )

    # Compare results
    for i in range(len(ref_pdf)):
        np.testing.assert_almost_equal(my_pdf[i], ref_pdf[i])


def test_generalized_gamma_compare_scipy_to_virocon_ICDF():
    """
    Test which compares the ICDF of a fitted scipy distribution and compare it 
    to the ICDF of a fitted virocon distribution. The results should be the 
    same since virocon uses scipy.

    """

    p = np.linspace(0.01, 0.99, num=100)

    # Create sample
    true_m = 5
    true_c = 2.42
    true_loc = 0
    true_lambda_ = 0.5
    gamma_samples = sts.gengamma.rvs(
        a=true_m,
        c=true_c,
        loc=true_loc,
        scale=1 / true_lambda_,
        size=100,
        random_state=42,
    )

    # Fit distribution with virocon
    my_gamma = GeneralizedGammaDistribution()
    my_gamma.fit(gamma_samples, method="mle")
    my_icdf = my_gamma.icdf(p)

    # Fit distribution with scipy
    ref_gamma = sts.gengamma.fit(gamma_samples, floc=0)
    ref_icdf = sts.gengamma.ppf(
        p, ref_gamma[0], ref_gamma[1], ref_gamma[2], ref_gamma[3]
    )

    # Compare results
    for i in range(len(ref_icdf)):
        np.testing.assert_almost_equal(my_icdf[i], ref_icdf[i])


def test_generalized_gamma_compare_scipy_fit_to_virocon_fit():
    """
    Test comparing the fitting of scipy to the fit of virocon. The results
    should be the same since virocon uses scipy.

    """

    # Create sample
    true_m = 5
    true_c = 2.42
    true_loc = 0
    true_lambda_ = 0.5
    gamma_samples = sts.gengamma.rvs(
        a=true_m,
        c=true_c,
        loc=true_loc,
        scale=1 / true_lambda_,
        size=100,
        random_state=42,
    )

    # Fit distribution with virocon
    my_gamma = GeneralizedGammaDistribution()
    my_gamma.fit(gamma_samples, method="mle")

    # Fit distribution with scipy
    ref_gamma = sts.gengamma.fit(gamma_samples, floc=0)

    # Compare results
    assert my_gamma.m == ref_gamma[0]
    assert my_gamma.c == ref_gamma[1]
    assert my_gamma._scale == ref_gamma[3]
