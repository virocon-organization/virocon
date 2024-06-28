# This module contains broad integration tests that are based on
# workflows how virocon is typically used.

# Test a workflow with the DNVGL sea state model (Hs, Tz)
import pytest

import numpy as np
import pandas as pd

from virocon import (
    GlobalHierarchicalModel,
    WeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    read_ec_benchmark_dataset,
    IFORMContour,
    ExponentiatedWeibullDistribution,
)

from virocon.distributions import LogNormalNormFitDistribution


@pytest.fixture(scope="module")
def dataset_dnvgl_hstz():
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")
    return data


@pytest.fixture(scope="module")
def refdata_dnvgl_hstz():
    with np.load(
        "tests/comparison-to-virocon-v1/reference_data/DNVGL/reference_data_DNVGL.npz"
    ) as npz_file:
        data_dict = dict(npz_file)

    data_keys = [
        "ref_f_weibull",
        "ref_weibull_params",
        "ref_givens",
        "ref_f_lognorm",
        "ref_mus",
        "ref_sigmas",
    ]
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


@pytest.mark.xfail
def test_DNVGL_Hs_Tz_model(dataset_dnvgl_hstz, refdata_dnvgl_hstz):
    # A 3-parameter power function (a dependence function).
    def _power3(x, a, b, c):
        return a + b * x**c

    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)

    x, dx = np.linspace([0.1, 0.1], [6, 22], num=100, retstep=True)

    dist_description_0 = {
        "distribution": WeibullDistribution(),
        "intervals": WidthOfIntervalSlicer(width=0.5),
    }
    dist_description_1 = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3},
    }
    ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])
    ghm.fit(dataset_dnvgl_hstz)
    f_weibull = ghm.distributions[0].pdf(x[:, 0])
    weibull_params = (
        ghm.distributions[0].beta,
        ghm.distributions[0].gamma,
        ghm.distributions[0].alpha,
    )

    lognorm = ghm.distributions[1]
    intervals = lognorm.data_intervals
    givens = lognorm.conditioning_values
    f_lognorm = []
    for given in givens:
        f_lognorm.append(lognorm.pdf(x[:, 1], given))

    f_lognorm = np.stack(f_lognorm, axis=1)
    mus = np.array([par["mu"] for par in lognorm.parameters_per_interval])
    sigmas = np.array([par["sigma"] for par in lognorm.parameters_per_interval])

    ref_f_weibull = refdata_dnvgl_hstz["ref_f_weibull"]
    ref_weibull_params = refdata_dnvgl_hstz["ref_weibull_params"]
    ref_intervals = 11
    ref_givens = refdata_dnvgl_hstz["ref_givens"]
    ref_f_lognorm = refdata_dnvgl_hstz["ref_f_lognorm"]
    ref_mus = refdata_dnvgl_hstz["ref_mus"]
    ref_sigmas = refdata_dnvgl_hstz["ref_sigmas"]

    assert len(intervals) == len(ref_intervals)
    for i in range(len(ref_intervals)):
        assert sorted(intervals[i]) == sorted(ref_intervals[i])

    np.testing.assert_allclose(f_weibull, ref_f_weibull)
    np.testing.assert_allclose(weibull_params, ref_weibull_params)
    np.testing.assert_allclose(givens, ref_givens)
    np.testing.assert_allclose(f_lognorm, ref_f_lognorm, rtol=1e-5)
    np.testing.assert_allclose(mus, ref_mus)
    np.testing.assert_allclose(sigmas, ref_sigmas)


# Test a work flow with the OMAE2020 wind speed - significant wave height model


@pytest.fixture(scope="module")
def dataset_omae2020_vhs():
    data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")
    data.columns = ["Datetime", "V", "Hs"]
    data = data[["V", "Hs"]]
    return data


@pytest.fixture(scope="module")
def refdata_omae2020_vhs():
    with np.load(
        "tests/comparison-to-virocon-v1/reference_data/OMAE2020/reference_data_OMAE2020.npz"
    ) as data:
        data_dict = dict(data)

    data_keys = [
        "ref_expweib0_params",
        "ref_f_expweib0",
        "ref_givens",
        "ref_alphas",
        "ref_betas",
        "ref_f_expweib1",
    ]
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


@pytest.mark.xfail
def test_OMAE2020(dataset_omae2020_vhs, refdata_omae2020_vhs):
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

    dist_description_vs = {
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

    ghm = GlobalHierarchicalModel([dist_description_vs, dist_description_hs])

    fit_description_vs = {"method": "wlsq", "weights": "quadratic"}
    fit_description_hs = {"method": "wlsq", "weights": "quadratic"}

    ghm.fit(dataset_omae2020_vhs, [fit_description_vs, fit_description_hs])

    x = np.linspace([0.1, 0.1], [30, 12], num=100)

    my_f_expweib0 = ghm.distributions[0].pdf(x[:, 0])
    my_expweib0_params = (
        ghm.distributions[0].alpha,
        ghm.distributions[0].beta,
        ghm.distributions[0].delta,
    )

    my_expweib1 = ghm.distributions[1]
    my_givens = my_expweib1.conditioning_values
    my_f_expweib1 = []
    for given in my_givens:
        my_f_expweib1.append(my_expweib1.pdf(x[:, 1], given))

    my_f_expweib1 = np.stack(my_f_expweib1, axis=1)

    my_alphas = np.array([par["alpha"] for par in my_expweib1.parameters_per_interval])
    my_betas = np.array([par["beta"] for par in my_expweib1.parameters_per_interval])
    my_intervals = my_expweib1.data_intervals

    ref_expweib0_params = refdata_omae2020_vhs["ref_expweib0_params"]
    ref_f_expweib0 = refdata_omae2020_vhs["ref_f_expweib0"]
    ref_intervals = refdata_omae2020_vhs["ref_intervals"]
    ref_givens = refdata_omae2020_vhs["ref_givens"]
    ref_alphas = refdata_omae2020_vhs["ref_alphas"]
    ref_betas = refdata_omae2020_vhs["ref_betas"]
    ref_f_expweib1 = refdata_omae2020_vhs["ref_f_expweib1"]

    np.testing.assert_almost_equal(my_expweib0_params, ref_expweib0_params)
    np.testing.assert_almost_equal(my_f_expweib0, ref_f_expweib0)
    for my_interval, ref_interval in zip(my_intervals, ref_intervals):
        np.testing.assert_almost_equal(np.sort(my_interval), np.sort(ref_interval))
    np.testing.assert_almost_equal(my_givens, ref_givens)
    np.testing.assert_almost_equal(my_alphas, ref_alphas)
    np.testing.assert_almost_equal(my_betas, ref_betas)
    np.testing.assert_almost_equal(my_f_expweib1, ref_f_expweib1)


# Test a workflow with the wind speed - turbluence intensity presented in Wind Energy Science


@pytest.fixture(scope="module")
def dataset_wes_sigmau():
    data = pd.read_csv("datasets/WES4_sample.csv", index_col="time")
    data.index = pd.to_timedelta(data.index)
    return data


@pytest.fixture(scope="module")
def refdata_wes_sigmau():
    with np.load(
        "tests/comparison-to-virocon-v1/reference_data/WES4/reference_data_WES4.npz"
    ) as npz_file:
        data_dict = dict(npz_file)

    data_keys = [
        "ref_weib_param",
        "ref_f_weib",
        "ref_givens",
        "ref_mu_norms",
        "ref_sigma_norms",
        "ref_mus",
        "ref_sigmas",
        "ref_f_ln",
        "ref_coordinates",
    ]

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


@pytest.mark.skip(reason="currently fails, needs to be checked in detail")
def test_WES4(dataset_wes_sigmau, refdata_wes_sigmau):
    # https://doi.org/10.5194/wes-4-325-2019

    class MyIntervalSlicer(WidthOfIntervalSlicer):
        def _slice(self, data):
            interval_slices, interval_references, interval_boundaries = super()._slice(
                data
            )

            # discard slices below 4 m/s
            ok_slices = []
            ok_references = []
            ok_boundaries = []
            for slice_, reference, boundaries in zip(
                interval_slices, interval_references, interval_boundaries
            ):
                if reference >= 4:
                    ok_slices.append(slice_)
                    ok_references.append(reference)
                    ok_boundaries.append(boundaries)

            return ok_slices, ok_references, ok_boundaries

    def _poly3(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    def _poly2(x, a, b, c):
        return a * x**2 + b * x + c

    poly3 = DependenceFunction(_poly3)
    poly2 = DependenceFunction(_poly2)

    dim0_description = {
        "distribution": WeibullDistribution(),
        "intervals": MyIntervalSlicer(width=1, reference="left", min_n_points=5),
    }

    dim1_description = {
        "distribution": LogNormalNormFitDistribution(),
        "conditional_on": 0,
        "parameters": {"mu_norm": poly3, "sigma_norm": poly2},
    }

    ghm = GlobalHierarchicalModel([dim0_description, dim1_description])
    ghm.fit(dataset_wes_sigmau)

    alpha = 1 / (5 * len(dataset_wes_sigmau))
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
    my_sigma_norms = np.array(
        [par["sigma_norm"] for par in my_ln.parameters_per_interval]
    )
    my_sigmas = [dist.sigma for dist in my_ln.distributions_per_interval]
    my_mus = [dist.mu for dist in my_ln.distributions_per_interval]

    ref_weib_param = refdata_wes_sigmau["ref_weib_param"]
    ref_f_weib = refdata_wes_sigmau["ref_f_weib"]
    ref_intervals = refdata_wes_sigmau["ref_intervals"]
    ref_givens = refdata_wes_sigmau["ref_givens"]
    ref_mu_norms = refdata_wes_sigmau["ref_mu_norms"]
    ref_sigma_norms = refdata_wes_sigmau["ref_sigma_norms"]
    ref_mus = refdata_wes_sigmau["ref_mus"]
    ref_sigmas = refdata_wes_sigmau["ref_sigmas"]
    ref_f_ln = refdata_wes_sigmau["ref_f_ln"]
    ref_coordinates = refdata_wes_sigmau["ref_coordinates"]

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
