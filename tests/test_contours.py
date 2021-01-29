import pytest
import numpy as np

from virocon.contours import IFORMContour, calculate_alpha
from virocon.dependencies import DependenceFunction
from virocon.distributions import (ExponentiatedWeibullDistribution, 
                                   LogNormalDistribution)
from virocon.models import GlobalHierarchicalModel

@pytest.fixture(scope="module")
def reference_coordinates_IFORM():
    with np.load("tests/reference_data/contours/reference_data_IFORM.npz") as data:
        ref_coordinates = data["ref_coordinates"]
    return ref_coordinates



def test_IFORM(reference_coordinates_IFORM):
    
    # Logarithmic square function.
    def _lnsquare2(x, a=3.62, b=5.77):
        return np.log(a + b * np.sqrt(x / 9.81))
    
    # 3-parameter function that asymptotically decreases (a dependence function).
    def _asymdecrease3(x, a=0, b=0.324, c=0.404):
        return a + b / (1 + c * x)
    
    lnsquare2 = DependenceFunction(_lnsquare2)
    asymdecrease3 = DependenceFunction(_asymdecrease3)
    
    dist_description_0 = {"distribution" : ExponentiatedWeibullDistribution,
                          "parameters" : {"alpha" : 0.207,
                                          "beta" : 0.684,
                                          "delta" : 7.79}
                          }
    
    dist_description_1 = {"distribution" : LogNormalDistribution,
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
    
    