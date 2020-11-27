import pytest

import numpy as np

from virocon.distributions import ExponentiatedWeibullDistribution



@pytest.fixture(scope="module")
def exp_weibull_reference_data():
    data = np.load("tests/reference_data/distributions/reference_data_exp_weibull.npz")
    return data



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
