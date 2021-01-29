import pytest
import numpy as np
import pandas as pd

from virocon.dependencies import DependenceFunction
from virocon.distributions import WeibullDistribution, LogNormalNormFitDistribution
from virocon.models import GlobalHierarchicalModel
from virocon.contours import IFORMContour
from virocon.intervals import WidthOfIntervalSlicer

@pytest.fixture(scope="module")
def dataset():
    data = pd.read_csv("datasets/WES4_sample.csv", index_col="time")
    data.index = pd.to_timedelta(data.index)
    return data

@pytest.fixture(scope="module")
def reference_data():
    with np.load("tests/reference_data/WES4/reference_data_WES4.npz") as npz_file:
        data_dict = dict(npz_file)
        
    data_keys = ["ref_weib_param", "ref_f_weib", "ref_givens", "ref_mu_norms", 
                 "ref_sigma_norms", "ref_mus", "ref_sigmas", "ref_f_ln", 
                 "ref_coordinates"]
    
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



def test_WES4(dataset, reference_data):
    #https://doi.org/10.5194/wes-4-325-2019
    
    class MyIntervalSlicer(WidthOfIntervalSlicer):
    
        def _slice(self, data):
            
            interval_slices, interval_centers = super()._slice(data)
            
            #discard slices below 4 m/s
            ok_slices = []
            ok_centers = []
            for slice_, center in zip(interval_slices, interval_centers):
                if center >=4:
                    ok_slices.append(slice_)
                    ok_centers.append(center)
            
            return ok_slices, ok_centers
        
    def _poly3(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d
    
    def _poly2(x, a, b, c):
        return a * x ** 2 + b * x + c
    
    poly3 = DependenceFunction(_poly3)
    poly2 = DependenceFunction(_poly2)
    
    dim0_description = {"distribution" : WeibullDistribution,
                        "intervals" : MyIntervalSlicer(width=1, min_n_points=5),
                        }
    
    dim1_description = {"distribution" : LogNormalNormFitDistribution,
                        "conditional_on" : 0,
                        "parameters" : {"mu_norm": poly3,
                                        "sigma_norm" : poly2},
                        }
    
    ghm = GlobalHierarchicalModel([dim0_description, dim1_description])
    ghm.fit(dataset)
    
    alpha = 1 / (5 * len(dataset))
    iform = IFORMContour(ghm, alpha)
    my_coordinates = iform.coordinates
    
    

    x_U = np.linspace(2, 40, num=100)
    x_sigma = np.linspace(0.02, 3.6, num=100)
    
    U_dist = ghm.distributions[0]
    my_weib_param = list(U_dist.parameters.values())
    my_f_weib = U_dist.pdf(x_U)
    
    my_ln = ghm.distributions[1]
    my_intervals = my_ln.data_intervals
    my_givens = my_ln.conditioning_values
    my_f_ln = []
    for given in my_givens:
        my_f_ln.append(my_ln.pdf(x_sigma, given))

    my_f_ln = np.stack(my_f_ln, axis=1)
    
    my_mu_norms = np.array([par["mu_norm"] for par in my_ln.parameters_per_interval])
    my_sigma_norms = np.array([par["sigma_norm"] for par in my_ln.parameters_per_interval])
    my_intervals = my_ln.data_intervals
    my_sigmas = [dist.sigma for dist in my_ln.distributions_per_interval]
    my_mus = [dist.mu for dist in my_ln.distributions_per_interval]
    
    
    
    ref_weib_param = reference_data["ref_weib_param"]
    ref_f_weib = reference_data["ref_f_weib"]
    ref_intervals = reference_data["ref_intervals"]
    ref_givens = reference_data["ref_givens"]
    ref_mu_norms = reference_data["ref_mu_norms"]
    ref_sigma_norms = reference_data["ref_sigma_norms"]
    ref_mus = reference_data["ref_mus"]
    ref_sigmas = reference_data["ref_sigmas"]
    ref_f_ln = reference_data["ref_f_ln"]
    ref_coordinates = reference_data["ref_coordinates"]


    np.testing.assert_allclose(my_weib_param, ref_weib_param)
    np.testing.assert_allclose(my_f_weib, ref_f_weib)
    
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(ref_intervals)):
        assert sorted(my_intervals[i]) == sorted(ref_intervals[i])
    
    np.testing.assert_allclose(my_givens, ref_givens)
    np.testing.assert_allclose(my_mu_norms, ref_mu_norms)
    np.testing.assert_allclose(my_sigma_norms, ref_sigma_norms)
    np.testing.assert_allclose(my_mus, ref_mus)
    np.testing.assert_allclose(my_sigmas, ref_sigmas)
    np.testing.assert_allclose(my_f_ln, ref_f_ln)
    np.testing.assert_allclose(my_coordinates, ref_coordinates)

    
    
