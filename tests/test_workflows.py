import pytest

import numpy as np
import pandas as pd

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    WeibullDistribution,
    ExponentiatedWeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    IFORMContour,
    HighestDensityContour,
    calculate_alpha,
    plot_marginal_quantiles,
    plot_dependence_functions,
    plot_2D_isodensity,
    plot_2D_contour,
)


def test_hs_tz_iform_contour():
    """
    Use a sea state dataset with the variables Hs and Tz,
    fit the join distribution recommended in DNVGL-RP-C203 to
    it and compute an IFORM contour. This tests reproduces
    the results published in Haseltseiner et al. (2019).

    Such a work flow is for example typical in ship design.

    Haselsteiner, A. F., Coe, R. G., Manuel, L., Nguyen, P. T. T.,
    Martin, N., & Eckert-Gallup, A. (2019). A benchmarking exercise
    on estimating extreme environmental conditions: Methodology &
    baseline results. Proc. 38th International Conference on Ocean,
    Offshore and Arctic Engineering (OMAE 2019).
    https://doi.org/10.1115/OMAE2019-96523

    DNV GL. (2017). Recommended practice DNVGL-RP-C205:
    Environmental conditions and environmental loads.
    """

    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

    # A 3-parameter power function (a dependence function).
    def _power3(x, a, b, c):
        return a + b * x**c

    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)

    dist_description_0 = {
        "distribution": WeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5),
    }
    dist_description_1 = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3},
    }
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    model.fit(data)

    axs = plot_marginal_quantiles(model, data)
    axs = plot_dependence_functions(model)
    ax = plot_2D_isodensity(model, data)

    alpha = calculate_alpha(1, 20)
    contour = IFORMContour(model, alpha)

    coordinates = contour.coordinates
    np.testing.assert_allclose(max(coordinates[:, 0]), 5.0, atol=0.5)
    np.testing.assert_allclose(max(coordinates[:, 1]), 16.1, atol=0.5)

    ax = plot_2D_contour(contour, sample=data)


def test_v_hs_hd_contour():
    """
    Use a wind speed - wave height dataset, fit the joint
    distribution that was proposed by Haselsteiner et al. (2020)
    and compute a highest density contour. This test reproduces
    the results presented in Haselestiner et al. (2020). The
    coorindates are availble at https://github.com/ec-benchmark-organizers/
    ec-benchmark/blob/master/results/exercise-1/contribution-4/haselsteiner_
    andreas_dataset_d_50.txt

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

    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))

    def _alpha3(x, a, b, c, d_of_x):
        return (a + b * x**c) / 2.0445 ** (1 / d_of_x(x))

    logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]

    alpha_bounds = [(0, None), (0, None), (None, None)]

    beta_dep = DependenceFunction(_logistics4, logistics_bounds, weights=lambda x, y: y)
    alpha_dep = DependenceFunction(
        _alpha3, alpha_bounds, d_of_x=beta_dep, weights=lambda x, y: y
    )

    dist_description_v = {
        "distribution": ExponentiatedWeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=2),
    }

    dist_description_hs = {
        "distribution": ExponentiatedWeibullDistribution(f_delta=5),
        "conditional_on": 0,
        "parameters": {
            "alpha": alpha_dep,
            "beta": beta_dep,
        },
    }

    model = GlobalHierarchicalModel([dist_description_v, dist_description_hs])

    fit_description_vs = {"method": "wlsq", "weights": "quadratic"}
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}

    model.fit(data, [fit_description_vs, fit_description_hs])

    axs = plot_marginal_quantiles(model, data)
    axs = plot_dependence_functions(model)
    ax = plot_2D_isodensity(model, data)

    alpha = calculate_alpha(1, 50)
    limits = [(0, 35), (0, 20)]
    contour = HighestDensityContour(model, alpha, limits=limits, deltas=[0.2, 0.2])

    coordinates = contour.coordinates
    np.testing.assert_allclose(max(coordinates[:, 0]), 29.9, atol=0.2)
    np.testing.assert_allclose(max(coordinates[:, 1]), 15.5, atol=0.2)
    np.testing.assert_allclose(min(coordinates[:, 0]), 0, atol=0.1)
    np.testing.assert_allclose(min(coordinates[:, 1]), 0, atol=0.1)

    ax = plot_2D_contour(contour, sample=data)
