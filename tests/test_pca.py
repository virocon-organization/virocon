# -*- coding: utf-8 -*-
import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_allclose

from viroconcom.utility import PCA
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour

@pytest.mark.parametrize("buoy_num", [46022, 46023, 46025, 46050])
def test_IFORM_contour_with_PCA_like_EckertGallupp2016(buoy_num):
    # Test the IFORM contour with PCA preprocessed data
    # as described by Eckert-Gallup2016: https://doi.org/10.1016/j.oceaneng.2015.12.018.
    # and https://github.com/WEC-Sim/WDRT/blob/master/WDRT/ESSC.py
    buoy_data = pd.read_csv(f"datasets/NDBC_buoy_{buoy_num}.csv")
    pca = PCA(buoy_data[["Hs", "T"]].values)
    pca_components = pca.transformed
    
    Time_SS = 1  # Sea state duration (hrs)
    Time_R = 100  # Return periods (yrs) of interest
    
    dist_description_0 = {"name": "InverseGaussian",
                          "dependency": (None, None, None),
                          "points_per_interval": 250,
                          }
    dist_description_1 = {"name": "Normal", 
                          "dependency": (None, 0, 0),
                          "functions": (None, "poly1", "poly2"),
                          }

    my_fit = Fit((pca_components[:, 0], pca_components[:, 1]),
                 (dist_description_0, dist_description_1))

    iform_contour = IFormContour(my_fit.mul_var_dist, Time_R, Time_SS, 1000)
    
    pca_coords = np.array(iform_contour.coordinates).T
    coords = pca.inverse_transform(pca_coords, clip=True)

    # Use previously generated and manually checked data as reference
    ref = pd.read_csv(f"tests/testfiles/PCA_IFORM_buoy_{buoy_num}.csv")
    
    assert_allclose(coords, ref[["Hs", "T"]].values, rtol=0, atol=1e-3)
