import pytest

import numpy as np
import pandas as pd

from virocon import (GlobalHierarchicalModel, ExponentiatedWeibullDistribution, 
                     DependenceFunction, WidthOfIntervalSlicer)

@pytest.fixture(scope="module")
def dataset():
    data = pd.read_csv("datasets/OMAE2020_Dataset_D.txt", sep=";")
    data.columns = ["Datetime", "V", "Hs"]
    data = data[["V", "Hs"]]
    return data

@pytest.fixture(scope="module")
def reference_data():
    with np.load("tests/reference_data/OMAE2020/reference_data_OMAE2020.npz") as data:
        data_dict = dict(data)
        
    data_keys = ["ref_expweib0_params", "ref_f_expweib0", "ref_givens", 
                 "ref_alphas", "ref_betas", "ref_f_expweib1"]
    ref_data = {}
    for key in data_keys:
        ref_data[key] = data_dict[key]
    
    ref_intervals = []
    i = 0
    while f"ref_interval{i}" in data_dict:
        ref_intervals.append(data_dict[f"ref_interval{i}"])
        i += 1
        
    ref_data["ref_intervals"] = ref_intervals
    return ref_data

def test_OMAE2020(dataset, reference_data):
    
    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))
    
    def _alpha3(x, a, b, c, d_of_x):
        return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))

    logistics_bounds = [(0, None),
                        (0, None),
                        (None, 0),
                        (0, None)]
    
    alpha_bounds = [(0, None), 
                    (0, None), 
                    (None, None)]
    
    beta_dep = DependenceFunction(_logistics4, logistics_bounds, 
                                  weights=lambda x, y : y)
    alpha_dep = DependenceFunction(_alpha3, alpha_bounds, d_of_x=beta_dep, 
                                   weights=lambda x, y : y)

    dist_description_vs = {"distribution" : ExponentiatedWeibullDistribution(),
                           "intervals" : WidthOfIntervalSlicer(width=2, offset=True),
                           }
    
    dist_description_hs = {"distribution" : ExponentiatedWeibullDistribution(f_delta=5),
                           "conditional_on" : 0,
                           "parameters" : {"alpha" : alpha_dep,
                                           "beta": beta_dep,
                                           },
                           }

    ghm = GlobalHierarchicalModel([dist_description_vs, dist_description_hs])

    fit_description_vs = {"method" : "wlsq", "weights": "quadratic"}
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
    
    ghm.fit(dataset, [fit_description_vs, fit_description_hs])
    
    x = np.linspace([0.1, 0.1], [30, 12], num=100)
    
    my_f_expweib0 = ghm.distributions[0].pdf(x[:, 0])
    my_expweib0_params = (ghm.distributions[0].alpha, ghm.distributions[0].beta, ghm.distributions[0].delta)
    
    my_expweib1 = ghm.distributions[1]
    my_givens = my_expweib1.conditioning_values
    my_f_expweib1 = []
    for given in my_givens:
        my_f_expweib1.append(my_expweib1.pdf(x[:, 1], given))
    
    my_f_expweib1 = np.stack(my_f_expweib1, axis=1)
    
    my_alphas = np.array([par["alpha"] for par in my_expweib1.parameters_per_interval])
    my_betas = np.array([par["beta"] for par in my_expweib1.parameters_per_interval])
    my_intervals = my_expweib1.data_intervals
    

    ref_expweib0_params = reference_data["ref_expweib0_params"]
    ref_f_expweib0 = reference_data["ref_f_expweib0"]
    ref_intervals = reference_data["ref_intervals"]
    ref_givens = reference_data["ref_givens"]
    ref_alphas = reference_data["ref_alphas"]
    ref_betas = reference_data["ref_betas"]
    ref_f_expweib1 = reference_data["ref_f_expweib1"]

    np.testing.assert_almost_equal(my_expweib0_params, ref_expweib0_params)
    np.testing.assert_almost_equal(my_f_expweib0, ref_f_expweib0)
    for my_interval, ref_interval in zip(my_intervals, ref_intervals):
        np.testing.assert_almost_equal(np.sort(my_interval), np.sort(ref_interval))
    np.testing.assert_almost_equal(my_givens, ref_givens)
    np.testing.assert_almost_equal(my_alphas, ref_alphas)
    np.testing.assert_almost_equal(my_betas, ref_betas)
    np.testing.assert_almost_equal(my_f_expweib1, ref_f_expweib1)