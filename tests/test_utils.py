import pytest

import pandas as pd

from pathlib import Path

from virocon.utils import read_ec_benchmark_dataset, ROOT_DIR


def test_read_ec_benchmark_dataset():
    path = str(ROOT_DIR.joinpath(Path("datasets/ec-benchmark_dataset_A.txt")))
    data = read_ec_benchmark_dataset(path)
    assert isinstance(data, pd.DataFrame)
    assert (data.columns == ["significant wave height (m)", 
                             "zero-up-crossing period (s)"]
            ).all()
    assert isinstance(data.index, pd.DatetimeIndex)
    assert len(data) == 82805