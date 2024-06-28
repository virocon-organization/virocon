import numpy as np
import pandas as pd

from virocon import (
    get_DNVGL_Hs_Tz,
    get_DNVGL_Hs_U,
    get_Windmeier_EW_Hs_S,
    get_Nonzero_EW_Hs_S,
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    TransformedModel,
    IFORMContour,
    variable_transform,
)


def test_DNVGL_Hs_Tz():
    # Load sea state measurements.
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A_1year.txt")

    # Define the structure of the joint distribution model.
    dist_descriptions, fit_descriptions, semantics = get_DNVGL_Hs_Tz()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model and compute an IFORM contour.
    model.fit(data, fit_descriptions)
    tr = 20  # Return period in years.
    ts = 1  # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(model, alpha)
    coords = contour.coordinates

    # Compare with the results from contours presented in Haselsteiner et al. (2019),
    # https://doi.org/10.1115/OMAE2019-96523
    # Highest Hs value is ca. 5 m, highest Tz value is ca. 16.2 s.
    np.testing.assert_allclose(max(coords[:, 0]), 5, atol=1.5)
    np.testing.assert_allclose(max(coords[:, 1]), 16.2, atol=2)


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
    tr = 50  # Return period in years.
    ts = 1  # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(model, alpha)
    coords = contour.coordinates

    # Compare with the results from contours presented in Haselsteiner et al. (2019),
    # https://doi.org/10.1115/OMAE2019-96523
    # Highest Hs value is ca. 9.5 m, highest U value is ca. 28 m/s.
    np.testing.assert_allclose(max(coords[:, 0]), 9.5, atol=1)
    np.testing.assert_allclose(max(coords[:, 1]), 28, atol=3)


def test_windmeier_ew_model():
    """
    Test the sea state model proposed by Kai-Lukas Windmeier (DOI: 10.26092/elib/2181).
    This model is defined in Hs-steepness space such that a variable transformation
    to Hs-Tz space is necessary.
    """

    # Load sea state measurements.
    data_hs_tz = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_C_1year.txt")
    hs = data_hs_tz["significant wave height (m)"]
    tz = data_hs_tz["zero-up-crossing period (s)"]
    temp, steepness = variable_transform.hs_tz_to_hs_s(hs, tz)
    steepness.name = "steepness"
    data_hs_s = pd.concat([hs, steepness], axis=1)

    # Define the structure of the joint distribution model.
    (
        dist_descriptions,
        fit_descriptions,
        semantics,
        transformations,
    ) = get_Windmeier_EW_Hs_S()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model in Hs-S space.
    model.fit(data_hs_s, fit_descriptions)

    f = model.marginal_pdf(1.0, 0)
    assert f > 0.1
    assert f < 2

    p = model.marginal_cdf(1.0, 0)
    assert p > 0.1
    assert p < 0.8

    # Transform the fitted model to Hs-Tz space.
    t_model = TransformedModel(
        model,
        transformations["transform"],
        transformations["inverse"],
        transformations["jacobian"],
        precision_factor=0.2,  # Use low precision to speed up test.
        random_state=42,
    )

    assert "TransformedModel" in str(t_model)
    assert "ExponentiatedWeibullDistribution" in str(t_model)

    p = t_model.cdf([2, 3])
    assert p < 1

    p = t_model.empirical_cdf([2, 3])
    assert p < 1

    try:
        t_model.marginal_pdf(3, 1)
        assert False
    except NotImplementedError:
        assert True

    try:
        t_model.marginal_cdf(3, 1)
        assert False
    except NotImplementedError:
        assert True

    # Compute a contour.
    tr = 1  # Return period in years.
    ts = 1  # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(
        t_model, alpha, n_points=20
    )  # Use few points to speed up tests.
    coords = contour.coordinates

    # A test during development of this test.
    # import matplotlib.pyplot as plt
    # from virocon import plot_2D_contour, plot_2D_isodensity
    # plot_2D_contour(contour, data_hs_tz, semantics=semantics, swap_axis=True)
    # plot_2D_isodensity(t_model, data_hs_tz, semantics=semantics, swap_axis=True)
    # plt.show()

    # Reference values are from Windmeier's Master thesis, page 60, DOI: 10.26092/elib/2181
    # Highest Hs values of the contour should be roughly Hs = 7.2 m, Tz = 11 s.
    # Note that in the Master thesis contours for dataset A and B are incorrect.
    np.testing.assert_allclose(max(coords[:, 0]), 7.2, atol=1)
    np.testing.assert_allclose(max(coords[:, 1]), 11, atol=1.5)


def test_nonzero_ew_model():
    """
    Test the nonzero EW sea state model.
    This model is defined in Hs-steepness space such that a variable transformation
    to Hs-Tz space is necessary.
    """

    # Load sea state measurements.
    data_hs_tz = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_C_1year.txt")
    hs = data_hs_tz["significant wave height (m)"]
    tz = data_hs_tz["zero-up-crossing period (s)"]
    temp, steepness = variable_transform.hs_tz_to_hs_s(hs, tz)
    steepness.name = "steepness"
    data_hs_s = pd.concat([hs, steepness], axis=1)

    # Define the structure of the joint distribution model.
    (
        dist_descriptions,
        fit_descriptions,
        semantics,
        transformations,
    ) = get_Nonzero_EW_Hs_S()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model in Hs-S space.
    model.fit(data_hs_s, fit_descriptions)

    # Transform the fitted model to Hs-Tz space.
    t_model = TransformedModel(
        model,
        transformations["transform"],
        transformations["inverse"],
        transformations["jacobian"],
        precision_factor=0.2,  # Use low precision to speed up test.
        random_state=42,
    )

    # Compute a contour.
    tr = 1  # Return period in years.
    ts = 1  # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(
        t_model, alpha, n_points=50
    )  # Use few points to speed up tests.
    coords = contour.coordinates

    # A test during development of this test.
    # import matplotlib.pyplot as plt
    # from virocon import plot_2D_contour, plot_2D_isodensity
    # plot_2D_contour(contour, data_hs_tz, semantics=semantics, swap_axis=True)
    # plot_2D_isodensity(t_model, data_hs_tz, semantics=semantics, swap_axis=True)
    # plt.show()

    np.testing.assert_allclose(max(coords[:, 0]), 7.8, atol=1)
    np.testing.assert_allclose(max(coords[:, 1]), 11.9, atol=1.5)
