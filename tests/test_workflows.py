import pytest

import numpy as np
import pandas as pd

from virocon import (read_ec_benchmark_dataset, GlobalHierarchicalModel, 
                     WeibullDistribution, LogNormalDistribution, DependenceFunction, 
                     WidthOfIntervalSlicer, IFORMContour, HighestDensityContour, 
                     calculate_alpha, )

def test_hs_tz_iform_contour():
    """
    Use a sea state dataset with the variables Hs and Tz,
    fit the join distribution recommended in DNVGL-RP-C203 to 
    it and compute an IFORM contour.

    Such a work flow is for example typical when generating a
    50-year contour when a ship is designed.

    DNV GL. (2017). Recommended practice DNVGL-RP-C205: 
    Environmental conditions and environmental loads.
    """

    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")


    # A 3-parameter power function (a dependence function).
    def _power3(x, a, b, c):
        return a + b * x ** c
    
    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)
        
    dist_description_0 = {"distribution" : WeibullDistribution(),
                          "intervals" : WidthOfIntervalSlicer(width=0.5)
                          }
    dist_description_1 = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": power3,
                                          "sigma" : exp3},
                          }
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    model.fit(data)

    alpha = calculate_alpha(1, 50)
    contour = IFORMContour(model, alpha)


def test_v_hs_hd_contour():
    """
    Use a wind speed - wave height dataset, fit the joint 
    distribution that was proposed by Haselsteiner et al (2020)
    and compute a highest density contour.

    Such a work flow is for example typical when generationg 
    a 50-year contour for DLC 1.6 in the offshore wind standard
    IEC 61400-3-1.

    Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, K.-D. (2020). 
    Global hierarchical models for wind and wave contours: Physical 
    interpretations of the dependence functions. Proc. 39th International 
    Conference on Ocean, Offshore and Arctic Engineering (OMAE 2020). 
    https://doi.org/10.1115/OMAE2020-18668

    International Electrotechnical Commission. (2019). Wind energy 
    generation systems - Part 3-1: Design requirements for fixed 
    offshore wind turbines (IEC 61400-3-1).
    """

    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")