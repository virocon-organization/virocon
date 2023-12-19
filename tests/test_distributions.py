import pytest
import numpy as np
import scipy.stats as sts

from virocon import (
    WeibullDistribution,
    LogNormalDistribution,
    NormalDistribution,
    ExponentiatedWeibullDistribution,
    GeneralizedGammaDistribution,
    ScipyDistribution,
)


def test_scipydistribution():
    # Generalized Gamma Distribution
    generalized_gamma_name = "gengamma"
    generalized_gamma_dist = sts.gengamma

    sample_params = {"a": 1.5, "c": 2, "loc": 0, "scale": 0.5}
    n = 1000
    sample = generalized_gamma_dist.rvs(**sample_params, size=n, random_state=42)
    sample.sort()
    x = np.linspace(sample.min(), sample.max())
    p = np.linspace(0, 1, num=n)

    class GeneralizedGammaDistributionByName(ScipyDistribution):
        scipy_dist_name = generalized_gamma_name

    class GeneralizedGammaDistributionByDist(ScipyDistribution):
        scipy_dist = generalized_gamma_dist

    # test wrong parameter instantiation
    with pytest.raises(TypeError):
        gengamma_by_name = GeneralizedGammaDistributionByName(floc=0)
    with pytest.raises(TypeError):
        gengamma_by_dist = GeneralizedGammaDistributionByDist(floc=0)

    gengamma_by_name = GeneralizedGammaDistributionByName(f_loc=0)
    gengamma_by_dist = GeneralizedGammaDistributionByDist(f_loc=0)

    # fit
    gengamma_by_name.fit(sample)
    gengamma_by_dist.fit(sample)

    assert list(gengamma_by_name.parameters.keys()) == ["a", "c", "loc", "scale"]
    assert list(gengamma_by_dist.parameters.keys()) == ["a", "c", "loc", "scale"]

    np.testing.assert_allclose(
        gengamma_by_name.parameters["a"], sample_params["a"], rtol=0.1
    )
    np.testing.assert_allclose(
        gengamma_by_name.parameters["c"], sample_params["c"], rtol=0.1
    )
    assert gengamma_by_name.parameters["loc"] == 0.0
    np.testing.assert_allclose(
        gengamma_by_name.parameters["scale"], sample_params["scale"], rtol=0.1
    )

    np.testing.assert_allclose(
        gengamma_by_name.parameters["a"], sample_params["a"], rtol=0.1
    )
    np.testing.assert_allclose(
        gengamma_by_name.parameters["c"], sample_params["c"], rtol=0.1
    )
    assert gengamma_by_name.parameters["loc"] == 0.0
    np.testing.assert_allclose(
        gengamma_by_name.parameters["scale"], sample_params["scale"], rtol=0.1
    )

    # sample
    sample_mean = sample.mean()
    sample_var = sample.var()

    sample_by_name = gengamma_by_name.draw_sample(n, random_state=42)
    sample_by_dist = gengamma_by_dist.draw_sample(n, random_state=42)
    mean_by_name = sample_by_name.mean()
    var_by_name = sample_by_name.var()
    mean_by_dist = sample_by_dist.mean()
    var_by_dist = sample_by_dist.var()
    assert mean_by_name == mean_by_dist
    assert var_by_name == var_by_dist

    np.testing.assert_allclose(sample_mean, mean_by_name, rtol=0.1)
    np.testing.assert_allclose(sample_var, var_by_name, rtol=0.1)

    # pdf
    pdf_by_name = gengamma_by_name.pdf(x)
    pdf_by_dist = gengamma_by_dist.pdf(x)
    assert (pdf_by_name == pdf_by_dist).all()
    assert (
        generalized_gamma_dist.pdf(x, **gengamma_by_name.parameters) == pdf_by_name
    ).all()

    # cdf
    cdf_by_name = gengamma_by_name.cdf(x)
    cdf_by_dist = gengamma_by_dist.cdf(x)
    assert (cdf_by_name == cdf_by_dist).all()
    assert (
        generalized_gamma_dist.cdf(x, **gengamma_by_name.parameters) == cdf_by_name
    ).all()

    # icdf
    icdf_by_name = gengamma_by_name.icdf(p)
    icdf_by_dist = gengamma_by_dist.icdf(p)
    assert (icdf_by_name == icdf_by_dist).all()
    assert (
        generalized_gamma_dist.ppf(p, **gengamma_by_name.parameters) == icdf_by_name
    ).all()


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
    Test reproducing the fitting of Ochi (1992) and compare it to
    virocons implementation of the generalized gamma distribution. The results
    should be the same.

    """

    # Define dist with parameters from the distribution of Fig. 4b in
    # M.K. Ochi, New approach for estimating the severest sea state from
    # statistical data , Coast. Eng. Chapter 38 (1992)
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


@pytest.fixture(
    params=[
        WeibullDistribution,
        LogNormalDistribution,
        NormalDistribution,
        ExponentiatedWeibullDistribution,
        GeneralizedGammaDistribution,
    ]
)
def dist_cls(request):
    return request.param


def test_use_fixed_values_if_provided(dist_cls):
    # create distributions without arguments to get the parameter names
    dist = dist_cls()
    param_names = dist.parameters.keys()
    f_param_names = ["f_" + par_name for par_name in param_names]

    # choose arbitrary values, that should work for all distribution and should be different to the default
    ref_values = [0.42 * (i + 1) for i in range(len(f_param_names))]
    f_params = {
        f_param_name: val for f_param_name, val in zip(f_param_names, ref_values)
    }
    # create distribution with fixed parameters
    dist = dist_cls(**f_params)
    param_values = list(dist.parameters.values())
    # check that the distribution uses the fixed values
    for reference_value, actual_value in zip(ref_values, param_values):
        assert reference_value == pytest.approx(actual_value)


def test_draw_sample(dist_cls):
    dist = dist_cls(1, 1)
    n = 4
    sample = dist.draw_sample(n, random_state=42)
    assert len(sample) == n
