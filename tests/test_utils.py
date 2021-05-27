import pytest

import pandas as pd
import numpy as np

from pathlib import Path

from virocon.utils import (
    read_ec_benchmark_dataset,
    ROOT_DIR,
    sort_points_to_form_continuous_line,
)


def test_read_ec_benchmark_dataset():
    path = str(ROOT_DIR.joinpath(Path("datasets/ec-benchmark_dataset_A.txt")))
    data = read_ec_benchmark_dataset(path)
    assert isinstance(data, pd.DataFrame)
    assert (
        data.columns == ["significant wave height (m)", "zero-up-crossing period (s)"]
    ).all()
    assert isinstance(data.index, pd.DatetimeIndex)
    assert len(data) == 82805


def test_sort_points_to_form_continuous_line():
    phi = np.linspace(0, 2 * np.pi, num=10, endpoint=False)
    ref_x = np.cos(phi)
    ref_y = np.sin(phi)

    shuffle_idx = [5, 2, 0, 6, 9, 4, 1, 8, 3, 7]
    rand_x = ref_x[shuffle_idx]
    rand_y = ref_y[shuffle_idx]

    my_x, my_y = sort_points_to_form_continuous_line(
        rand_x, rand_y, search_for_optimal_start=True
    )

    # Clockwise or anti-clockwise order is not deterministic,
    # but both are okay.
    try:
        np.testing.assert_array_equal(my_x, ref_x)
        np.testing.assert_array_equal(my_y, ref_y)
    except AssertionError:
        np.testing.assert_array_equal(my_x[::-1], ref_x)
        np.testing.assert_array_equal(my_y[::-1], ref_y)
