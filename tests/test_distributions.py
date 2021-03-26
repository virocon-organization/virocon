import pytest

import numpy as np

import scipy.stats as sts

from virocon import ExponentiatedWeibullDistribution



@pytest.fixture(scope="module")
def exp_weibull_reference_data():
    with np.load("tests/reference_data/distributions/reference_data_exp_weibull.npz") as data:
        data_dict = dict(data)
    return data_dict

@pytest.fixture(scope="module")
def exp_weibull_reference_data_wlsq_fit():
    with np.load("tests/reference_data/distributions/reference_data_exp_weibull_wlsq_fit.npz") as data:
        data_dict = dict(data)
    return data_dict



def test_ExponentiatedWeibull_pdf_cdf_icdf(exp_weibull_reference_data):
    OMAE2020_param = {"alpha" : 10.0, 
                      "beta" : 2.42, 
                      "delta" : 0.761
                      }
    x = np.linspace(2, 15, num=100)
    p = np.linspace(0.01, 0.99, num=100)
    my_expweibull = ExponentiatedWeibullDistribution(**OMAE2020_param)
    my_pdf = my_expweibull.pdf(x)
    my_cdf = my_expweibull.cdf(x)
    my_icdf = my_expweibull.icdf(p)
    
    ref_pdf = exp_weibull_reference_data["ref_pdf"]
    ref_cdf = exp_weibull_reference_data["ref_cdf"]
    ref_icdf = exp_weibull_reference_data["ref_icdf"]
    
    np.testing.assert_allclose(my_pdf, ref_pdf)
    np.testing.assert_allclose(my_cdf, ref_cdf)
    np.testing.assert_allclose(my_icdf, ref_icdf)
    
    
def test_ExponentiatedWeibull_wlsq_fit(exp_weibull_reference_data_wlsq_fit):
    x = np.linspace(2, 15, num=100)
    p = np.linspace(0.01, 0.99, num=100)
    
    true_alpha = 10
    true_beta = 2.42
    true_delta = 0.761
    expweibull_samples = sts.exponweib.rvs(a=true_delta, c=true_beta, 
                                           loc=0, scale=true_alpha, 
                                           size=100, random_state=42)
    
    my_expweibull = ExponentiatedWeibullDistribution(fit_method="lsq", weights="quadratic")
    
    my_expweibull.fit(expweibull_samples)
    
    my_pdf = my_expweibull.pdf(x)
    my_cdf = my_expweibull.cdf(x)
    my_icdf = my_expweibull.icdf(p)
    my_alpha = my_expweibull.alpha
    my_beta = my_expweibull.beta
    my_delta = my_expweibull.delta
    
    ref_pdf = exp_weibull_reference_data_wlsq_fit["ref_pdf"]
    ref_cdf = exp_weibull_reference_data_wlsq_fit["ref_cdf"]
    ref_icdf = exp_weibull_reference_data_wlsq_fit["ref_icdf"]
    ref_alpha = exp_weibull_reference_data_wlsq_fit["ref_alpha"]
    ref_beta = exp_weibull_reference_data_wlsq_fit["ref_beta"]
    ref_delta = exp_weibull_reference_data_wlsq_fit["ref_delta"]
    
    np.testing.assert_allclose(my_alpha, ref_alpha)
    np.testing.assert_allclose(my_beta, ref_beta)
    np.testing.assert_allclose(my_delta, ref_delta)
    np.testing.assert_allclose(my_pdf, ref_pdf)
    np.testing.assert_allclose(my_cdf, ref_cdf)
    np.testing.assert_allclose(my_icdf, ref_icdf)
    
