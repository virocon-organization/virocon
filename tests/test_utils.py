import pytest

import pandas as pd
import numpy as np

from pathlib import Path

from virocon import (
    DependenceFunction,
    WeibullDistribution,
    LogNormalDistribution,
    GlobalHierarchicalModel,
    IFORMContour
)

from virocon.utils import (
    read_ec_benchmark_dataset,
    ROOT_DIR,
    sort_points_to_form_continuous_line,
    calculate_design_conditions
)


@pytest.fixture(scope="module")
def seastate_model():
    """
    This joint distribution model described by Vanem and Bitner-Gregersen (2012)
    is widely used in academia. Here, we use it for evaluation.
    DOI: 10.1016/j.apor.2012.05.006
    """

    def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x**c

    # A 3-parameter exponential function (a dependence function).
    def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
        return a + b * np.exp(c * x)

    bounds = [(0, None), (0, None), (None, None)]
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)

    dist_description_0 = {
        "distribution": WeibullDistribution(alpha=2.776, beta=1.471, gamma=0.8888),
    }
    dist_description_1 = {
        "distribution": LogNormalDistribution(),
        "conditional_on": 0,
        "parameters": {"mu": power3, "sigma": exp3},
    }
    model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

    return model


def test_read_ec_benchmark_dataset():
    path = str(ROOT_DIR.joinpath(Path("datasets/ec-benchmark_dataset_A.txt")))
    data = read_ec_benchmark_dataset(path)
    assert isinstance(data, pd.DataFrame)
    assert (
        data.columns == ["significant wave height (m)", "zero-up-crossing period (s)"]
    ).all()
    assert isinstance(data.index, pd.DatetimeIndex)
    assert len(data) == 82805


def test_calculate_design_conditions(seastate_model):
    # Test design condition calculation with  the IFORM contour presented in
    # Haselsteiner et al. (2017; DOI: 10.1016/j.coastaleng.2017.03.002 .
    alpha = 1 / (25 * 365.25 * 24/3)
    contour = IFORMContour(seastate_model, alpha)

    design_conditions = calculate_design_conditions(contour, steps=None, swap_axis=False)
    assert design_conditions.shape == (10, 2)
    np.testing.assert_allclose(design_conditions[1], [2.5, 11.6], atol=0.5)
    np.testing.assert_allclose(design_conditions[9], [15.2, 13.4], atol=0.5)

    steps = 20
    design_conditions = calculate_design_conditions(contour, steps=steps, swap_axis=False)
    assert design_conditions.shape == (steps, 2)
    np.testing.assert_allclose(design_conditions[1], [1.6, 10.9], atol=0.5)
    np.testing.assert_allclose(design_conditions[19], [15.2, 13.4], atol=0.5)

    design_conditions = calculate_design_conditions(contour, steps=steps, swap_axis=True)
    assert design_conditions.shape == (steps, 2)
    np.testing.assert_allclose(design_conditions[0], [2.6, 1.1], atol=0.5)
    np.testing.assert_allclose(design_conditions[19], [13.9, 14.1], atol=0.5)

def test_sort_points_to_form_continuous_line():
    phi = np.linspace(0, 1.8 * np.pi, num=10, endpoint=False)
    ref_x = np.cos(phi)
    ref_y = np.sin(phi)

    shuffle_idx = [5, 2, 0, 6, 9, 4, 1, 8, 3, 7]
    rand_x = ref_x[shuffle_idx]
    rand_y = ref_y[shuffle_idx]

    my_x, my_y = sort_points_to_form_continuous_line(
        rand_x, rand_y, search_for_optimal_start=True
    )

    # For some reason this test fails on MacOS (but works on Ubuntu and Windows).
    # In principal, clockwise and anti-clockwise are both correct solutions, but on MacOS
    # it is neither, the starting point is in the middle of the sequency.
    try:
        np.testing.assert_array_equal(my_x, ref_x)
        np.testing.assert_array_equal(my_y, ref_y)
    except AssertionError:
        np.testing.assert_array_equal(my_x[::-1], ref_x)
        np.testing.assert_array_equal(my_y[::-1], ref_y)
