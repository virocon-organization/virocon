import pytest
import numpy as np
import scipy.stats as sts

from virocon import ExponentiatedWeibullDistribution


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
