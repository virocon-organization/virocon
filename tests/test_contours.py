import pytest
import numpy as np

from virocon.contours import (IFORMContour, 
                              ISORMContour,
                              HighestDensityContour, 
                              DirectSamplingContour,
                              calculate_alpha)
from virocon.dependencies import DependenceFunction
from virocon.distributions import (ExponentiatedWeibullDistribution, 
                                   LogNormalDistribution, 
                                   WeibullDistribution)
from virocon.models import GlobalHierarchicalModel

@pytest.fixture(scope="module")
def reference_coordinates_IFORM():
    with np.load("tests/reference_data/contours/reference_data_IFORM.npz") as data:
        ref_coordinates = data["ref_coordinates"]
    return ref_coordinates


@pytest.fixture(scope="module")
def reference_coordinates_ISORM():
    with np.load("tests/reference_data/contours/reference_data_ISORM.npz") as data:
        ref_coordinates = data["ref_coordinates"]
    return ref_coordinates


@pytest.fixture(scope="module")
def reference_coordinates_HDC():
    with np.load("tests/reference_data/contours/reference_data_HDC.npz") as data:
        ref_coordinates = data["ref_coordinates"]
    return ref_coordinates


@pytest.fixture(scope="module")
def reference_data_DSContour():
    with np.load("tests/reference_data/contours/reference_data_DSContour.npz") as data:
        data_dict = dict(data)
    return data_dict


def test_IFORM(reference_coordinates_IFORM):
    
    # Logarithmic square function.
    def _lnsquare2(x, a=3.62, b=5.77):
        return np.log(a + b * np.sqrt(x / 9.81))
    
    # 3-parameter function that asymptotically decreases (a dependence function).
    def _asymdecrease3(x, a=0, b=0.324, c=0.404):
        return a + b / (1 + c * x)
    
    lnsquare2 = DependenceFunction(_lnsquare2)
    asymdecrease3 = DependenceFunction(_asymdecrease3)
    
    dist_description_0 = {"distribution" : ExponentiatedWeibullDistribution(alpha=0.207, beta=0.684, delta=7.79),
                          }
    
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": lnsquare2,
                                          "sigma" : asymdecrease3},
                          }
    
    ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    
    state_duration = 3
    return_period = 20
    alpha = calculate_alpha(state_duration, return_period)
    my_iform = IFORMContour(ghm, alpha)
    
    my_coordinates = my_iform.coordinates
    
    np.testing.assert_allclose(my_coordinates, reference_coordinates_IFORM)
    
    
def test_ISORM(reference_coordinates_ISORM):

    # Logarithmic square function.
    def _lnsquare2(x, a=3.62, b=5.77):
        return np.log(a + b * np.sqrt(x / 9.81))
    
    # 3-parameter function that asymptotically decreases (a dependence function).
    def _asymdecrease3(x, a=0, b=0.324, c=0.404):
        return a + b / (1 + c * x)
    
    
    lnsquare2 = DependenceFunction(_lnsquare2)
    asymdecrease3 = DependenceFunction(_asymdecrease3)
    
    dist_description_0 = {"distribution" : ExponentiatedWeibullDistribution(alpha=0.207,
                                                                            beta=0.684,
                                                                            delta=7.79),
                          }
    
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": lnsquare2,
                                          "sigma" : asymdecrease3},
                          }
    
    ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    
    
    state_duration = 3
    return_period = 20
    alpha = calculate_alpha(state_duration, return_period)
    my_isorm = ISORMContour(ghm, alpha)
    
    my_coordinates = my_isorm.coordinates
    
    np.testing.assert_allclose(my_coordinates, reference_coordinates_ISORM)
    
    
def test_HDC(reference_coordinates_HDC):
    
    def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x ** c
    
    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
        return a + b * np.exp(c * x)
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)
    
    dist_description_0 = {"distribution" : WeibullDistribution(lambda_=2.776,
                                                               k=1.471,
                                                               theta=0.8888),
                          }
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": power3,
                                          "sigma" : exp3},
                          }
    ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    
    alpha = calculate_alpha(3, 50)
    limits = [(0, 20), (0, 18)]
    deltas = [0.1, 0.1]
    my_contour = HighestDensityContour(ghm, alpha, limits, deltas)
    
    my_coordinates = my_contour.coordinates
    
    np.testing.assert_allclose(my_coordinates, reference_coordinates_HDC)
    

def test_DirectSamplingContour(reference_data_DSContour):
    
    sample = reference_data_DSContour["sample"]
    ref_coordinates = reference_data_DSContour["ref_coordinates"]
    
    def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x ** c
    
    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
        return a + b * np.exp(c * x)
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)
    
    dist_description_0 = {"distribution" : WeibullDistribution(lambda_=2.776,
                                                               k=1.471,
                                                               theta=0.8888),
                          }
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": power3,
                                          "sigma" : exp3},
                          }
    ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    
    alpha = calculate_alpha(3, 50)
    my_ds_contour = DirectSamplingContour(ghm, alpha, sample=sample)
    
    my_coordinates = my_ds_contour.coordinates
    
    np.testing.assert_allclose(my_coordinates, ref_coordinates)
    
    