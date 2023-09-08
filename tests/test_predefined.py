import numpy as np

from virocon import (
    get_DNVGL_Hs_Tz,
    get_DNVGL_Hs_U,
    get_Hs_S_ExpWeib_WLS_Hs_Tz,
    read_ec_benchmark_dataset,
    factor,
    GlobalHierarchicalModel,
    TransformedModel,
    IFORMContour,
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


def test_kai_ew_mode():
    """
    Test the EW sea state model proposed by Kai-Lukas Windmeier (DOI: 10.26092/elib/2181).
    This model uses a variable transformation.
    """

    # Load sea state measurements.
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A_1year.txt")

    # Define the structure of the joint distribution model.
    (
        dist_descriptions,
        fit_descriptions,
        semantics,
        transformations,
    ) = get_Hs_S_ExpWeib_WLS_Hs_Tz()
    model = GlobalHierarchicalModel(dist_descriptions)

    # Fit the model.
    model.fit(data, fit_descriptions)
    t_model = TransformedModel(
        model,
        transformations["transform"],
        transformations["inverse"],
        transformations["jacobian"],
        precision_factor=0.2,
        random_state=42,
    )

    # TODO: Speed this up (takes long due to the contour calculation which
    # uses a Monte Carlo based method.
    # Compute a contour.
    tr = 1  # Return period in years.
    ts = 1  # Sea state duration in hours.
    alpha = 1 / (tr * 365.25 * 24 / ts)
    contour = IFORMContour(t_model, alpha, n_points=10)
    coords = contour.coordinates

    print("coords: " + str(coords))

    # Reference values are from Kai's Master thesis, page 60, DOI: 10.26092/elib/2181
    # Highest Hs values of the contour should be roughly Hs = 7.2 m, Tz = 11 s.
    np.testing.assert_allclose(max(coords[:, 0]), 7.2, atol=1)
    np.testing.assert_allclose(max(coords[:, 1]), 11, atol=1.5)
