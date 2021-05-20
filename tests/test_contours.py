import pytest
import numpy as np

from virocon import (IFORMContour, ISORMContour, HighestDensityContour, 
                     DirectSamplingContour, calculate_alpha, DependenceFunction,
                     ExponentiatedWeibullDistribution, LogNormalDistribution, 
                     WeibullDistribution, GlobalHierarchicalModel)


@pytest.fixture(scope="module")
def seastate_model():
    """
    This joint distribution model described by Vanem and Bitner-Gregersen (2012)
    is widely used in academia. Here, we use it for evaluation. 
    DOI: 10.1016/j.apor.2012.05.006
    """

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
    
    dist_description_0 = {"distribution" : WeibullDistribution(alpha=2.776,
                                                               beta=1.471,
                                                               gamma=0.8888),
                          }
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": power3,
                                          "sigma" : exp3},
                          }
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

    return model
  
    
def test_DirectSamplingContour(seastate_model):
    """
    Computes a direct sampling contour and compares it with results
    from Huseby et al. (2013; DOI: 10.1016/j.oceaneng.2012.12.034, Tab. 5).
    """
    ref_contour_hs_1 = \
        [9.99, 10.65, 10.99, 11.25, 11.25, 11.41, 11.42, 11.46, 11.48,
            11.54, 11.57, 11.56, 11.58, 11.59, 11.59, 11.60, 11.60, 11.59,
            11.59, 11.56, 11.53, 11.46, 11.26, 10.88, 7.44, 2.05]
    ref_contour_tz_1 = \
        [12.34, 12.34, 12.31, 12.25, 12.25, 12.18, 12.17, 12.15, 12.13,
            12.06, 12.02, 12.03, 12.00, 11.96, 11.95, 11.86, 11.84, 11.77,
            11.76, 11.67, 11.60, 11.47, 11.20, 10.77, 7.68, 3.76]

    alpha = calculate_alpha(6, 1)

    prng = np.random.RandomState(42) # Fix the random seed for consistency.
    # Because a random sample is drawn (and fixing the random seed with 
    # .np.random.RandomState) does not work, results will be different each 
    # time the test is run. Sometimes the test might fail.
    my_ds_contour = DirectSamplingContour(seastate_model, alpha, n=500000, deg_step=6)
       
    my_coordinates = my_ds_contour.coordinates
    
    np.testing.assert_allclose(my_coordinates[0:26,0], ref_contour_hs_1, atol=0.75)
    np.testing.assert_allclose(my_coordinates[0:26,1], ref_contour_tz_1, atol=0.75)


def test_HighestDensityContour(seastate_model):
    """
    Compare the coordinates of a HD contour with the results from Haselsteiner 
    et al. (2017; DOI: 10.1016/j.coastaleng.2017.03.002, Fig. 5)
    """
    alpha = calculate_alpha(3, 25)
    limits = [(0, 20), (0, 18)]
    deltas = [0.05, 0.05]
    my_contour = HighestDensityContour(seastate_model, alpha, limits, deltas)
    
    my_coordinates = my_contour.coordinates
    np.testing.assert_allclose(max(my_coordinates[:,0]), 16.79, atol=0.05)
    np.testing.assert_allclose(max(my_coordinates[:,1]), 14.64, atol=0.05)
    
 
