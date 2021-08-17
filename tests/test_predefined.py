import pytest
import numpy as np

from virocon import (get_DNVGL_Hs_Tz, get_DNVGL_Hs_U, read_ec_benchmark_dataset, 
    GlobalHierarchicalModel, IFORMContour)


def test_DNVGL_Hs_Tz():
    # Load sea state measurements.
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A_1year.txt")

    # Define the structure of the joint distribution model.
    dist_descriptions, fit_descriptions, semantics = get_DNVGL_Hs_Tz()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model and compute an IFORM contour.
    model.fit(data, fit_descriptions)
    tr = 20 # Return period in years.
    ts = 1 # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(model, alpha)
    coords = contour.coordinates

    # Compare with the results from contours presented inHaselsteiner et al. (2019), 
    # https://doi.org/10.1115/OMAE2019-96523
    # Highest Hs value is ca. 5 m, highest Tz value is ca. 16.2 s.
    np.testing.assert_allclose(max(coords[:,0]), 5, atol=1.5)
    np.testing.assert_allclose(max(coords[:,1]), 16.2, atol=2)


def test_DNVGL_Hs_U():
    # Load sea state measurements.
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D_1year.txt")

    # Switch the order of wind speed and significant wave height in the data frame
    # because the DNVGL model requires the order Hs-U and the original dataset is U-Hs.
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:1]
    data = data[cols]

    # Define the structure of the joint distribution model.
    dist_descriptions, fit_descriptions, semantics = get_DNVGL_Hs_U()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model and compute an IFORM contour.
    model.fit(data, fit_descriptions)
    tr = 50 # Return period in years.
    ts = 1 # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(model, alpha)
    coords = contour.coordinates

    # Compare with the results from contours presented inHaselsteiner et al. (2019), 
    # https://doi.org/10.1115/OMAE2019-96523
    # Highest Hs value is ca. 9.5 m, highest U value is ca. 28 m/s.
    np.testing.assert_allclose(max(coords[:,0]), 9.5, atol=1)
    np.testing.assert_allclose(max(coords[:,1]), 28, atol=3)
