import pytest

import numpy as np

from virocon import (
    WidthOfIntervalSlicer,
    NumberOfIntervalsSlicer,
    PointsPerIntervalSlicer,
)


@pytest.fixture(scope="module")
def test_data():
    """
    Function to create test data.

    Returns
    -------
    Numpy array
        Test data.

    """

    return np.array([1.2, 1.5, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0])


@pytest.fixture(
    params=[
        (
            {"width": 1, "reference": "left", "min_n_points": 1},
            [1, 2, 3, 4, 5],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
        ),
        (
            {"width": 1, "reference": "left", "right_open": False, "min_n_points": 1},
            [1, 2, 3, 4],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        ),
        (
            {"width": 1, "min_n_points": 1},
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
        ),
        (
            {"width": 1, "right_open": False, "min_n_points": 1},
            [1.5, 2.5, 3.5, 4.5],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]],
            [[1, 2], [2, 3], [3, 4], [4, 5]],
        ),
        (
            {"width": 1, "reference": np.median, "min_n_points": 1},
            [1.35, 2.5, 3.5, 4.0, 5.0],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
        ),
        (
            {"width": 1, "reference": "left", "min_n_points": 2},
            [1, 2, 3],
            [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        ),
    ]
)
def woi_params_and_ref(request):
    """
    Fixture that applies the WidthOfIntervalSlicer to the prepared test case.

    Parameters
    ----------
    request : tuple
        Prepared test case for WidthOfIntervalSlicer.

    Returns
    -------
    dict
        Characteristics of interval.

    """

    params = request.param[0]
    references = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {
        "params": params,
        "references": references,
        "intervals": intervals,
        "boundaries": boundaries,
    }


@pytest.fixture(
    params=[
        (
            {"n_intervals": 2, "min_n_points": 1, "min_n_intervals": 2},
            [2.15, 4.05],
            [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]],
            [[1.2, 3.1], [3.1, 5.0]],
        ),
        (
            {"n_intervals": 2, "include_max": False, "min_n_points": 1,},
            [2.15, 4.05],
            [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0]],
            [[1.2, 3.1], [3.1, 5.0]],
        ),
        (
            {"n_intervals": 3, "min_n_points": 1},
            [1.8333333333333333, 3.0999999999999996, 4.366666666666666],
            [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0, 5.0]],
            [
                [1.2, 2.466666666666667],
                [2.4666666666666663, 3.733333333333333],
                [3.733333333333333, 5.0],
            ],
        ),
        (
            {"n_intervals": 3, "min_n_points": 3, "min_n_intervals": 2},
            [1.8333333333333333, 3.0999999999999996],
            [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]],
            [[1.2, 2.466666666666667], [2.4666666666666663, 3.733333333333333]],
        ),
        (
            {"n_intervals": 3, "include_max": False, "min_n_points": 1},
            [1.8333333333333333, 3.0999999999999996, 4.366666666666666],
            [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0]],
            [
                [1.2, 2.466666666666667],
                [2.4666666666666663, 3.733333333333333],
                [3.733333333333333, 5.0],
            ],
        ),
        (
            {"n_intervals": 2, "value_range": (0, 5), "min_n_points": 1,},
            [1.25, 3.75],
            [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0]],
            [[0, 2.5], [2.5, 5.0]],
        ),
        (
            {"n_intervals": 2, "reference": np.median, "min_n_points": 1,},
            [2.4, 3.6],
            [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]],
            [[1.2, 3.1], [3.1, 5.0]],
        ),
        (
            {
                "n_intervals": 3,
                "min_n_points": 3,
            },  # should raise RuntimeError (too few intervals)
            None,
            None,
            None,
        ),
    ]
)
def noi_params_and_ref(request):
    """
    Fixture that applies the NumberOfIntervalsSlicer to the prepared test case.

    Parameters
    ----------
    request : tuple
        Prepared test case for NumberOfIntervalsSlicer.

    Returns
    -------
    dict
        Characteristics of interval.

    """

    params = request.param[0]
    references = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {
        "params": params,
        "references": references,
        "intervals": intervals,
        "boundaries": boundaries,
    }


@pytest.fixture(
    params=[
        (
            {"n_points": 2, "min_n_points": 1},
            [1.35, 2.45, 2.85, 3.55, 4.5],
            [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1], [3.5, 3.6], [4.0, 5.0]],
            [[1.2, 1.95], [1.95, 2.55], [2.55, 3.3], [3.3, 3.8], [3.8, 5.0]],
        ),
        (
            {"n_points": 3, "min_n_points": 1},
            [1.2, 2.4, 3.1, 4.0],
            [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
            [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]],
        ),
        (
            {"n_points": 3, "last_full": False, "min_n_points": 1},
            [1.5, 2.6, 3.6, 5.0],
            [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]],
            [[1.2, 2.45], [2.45, 3.3], [3.3, 4.5], [4.5, 5.0]],
        ),
        (
            {"n_points": 3,},
            [2.4, 3.1, 4.0],
            [[1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
            [[1.5, 2.55], [2.55, 3.55], [3.55, 5.0]],
        ),
        (
            {"n_points": 3, "reference": np.mean, "min_n_points": 1},
            [1.2, 2.1333333333333333, 3.0666666666666664, 4.2],
            [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
            [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]],
        ),
    ]
)
def ppi_params_and_ref(request):
    """
    Fixture that applies the PointsPerIntervalSlicer to the prepared test case.

    Parameters
    ----------
    request : tuple
        Prepared test case for PointsPerIntervalSlicer.

    Returns
    -------
    dict
        Characteristics of interval.

    """

    params = request.param[0]
    references = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {
        "params": params,
        "references": references,
        "intervals": intervals,
        "boundaries": boundaries,
    }


def test_width_of_interval_slicer(test_data, woi_params_and_ref):
    """
    Function to compare characterisics of reference interval generated by
    WidthOfIntervalSlicer and manually generated interval.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
    woi_params_and_ref : tuple
        Reference interval generated by the WidthOfIntervalSlicer funciton.

    Returns
    -------
    None.

    """

    params = woi_params_and_ref["params"]
    ref_references = woi_params_and_ref["references"]
    ref_intervals = woi_params_and_ref["intervals"]
    ref_boundaries = woi_params_and_ref["boundaries"]
    slicer = WidthOfIntervalSlicer(**params)

    my_slices, my_references, my_boundaries = slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_intervals_slicer(test_data, noi_params_and_ref):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalsSlicer and manually generated interval.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
    noi_params_and_ref : tuple
        Reference interval generated by the NumberOfIntervalsSlicer funciton.

    Returns
    -------
    None.

    """

    params = noi_params_and_ref["params"]
    ref_references = noi_params_and_ref["references"]
    ref_intervals = noi_params_and_ref["intervals"]
    ref_boundaries = noi_params_and_ref["boundaries"]
    slicer = NumberOfIntervalsSlicer(**params)

    if ref_references is None:
        with pytest.raises(RuntimeError):
            my_slices, my_references, my_boundaries = slicer.slice_(test_data)
    else:
        my_slices, my_references, my_boundaries = slicer.slice_(test_data)
        my_intervals = [test_data[slice_] for slice_ in my_slices]
        np.testing.assert_almost_equal(my_references, ref_references)
        assert len(my_intervals) == len(ref_intervals)
        for i in range(len(my_intervals)):
            np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
        for i in range(len(my_boundaries)):
            np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer(test_data, ppi_params_and_ref):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
    ppi_params_and_ref : tuple
        Reference interval generated by the PointsPerIntervalSlicer funciton.

    Returns
    -------
    None.

    """

    params = ppi_params_and_ref["params"]
    ref_references = ppi_params_and_ref["references"]
    ref_intervals = ppi_params_and_ref["intervals"]
    ref_boundaries = ppi_params_and_ref["boundaries"]
    slicer = PointsPerIntervalSlicer(**params)

    my_slices, my_references, my_boundaries = slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer_npoints2(test_data):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval..

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n_points= 2
    ref_intervals = [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1], [3.5, 3.6], [4.0, 5.0]]
    ref_references = [np.median(inter) for inter in ref_intervals]
    ref_boundaries = [[1.2, 1.95], [1.95, 2.55], [2.55, 3.3], [3.3, 3.8], [3.8, 5.0]]
    n_points_slicer = PointsPerIntervalSlicer(2, min_n_points=1)
    my_slices, my_references, my_boundaries = n_points_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer_npoints3_lastfull(test_data):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval..

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n_points= 3, last_full=True
    ref_intervals = [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
    ref_references = [np.median(inter) for inter in ref_intervals]
    ref_boundaries = [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]]
    n_points_slicer = PointsPerIntervalSlicer(3, last_full=True, min_n_points=1)
    my_slices, my_references, my_boundaries = n_points_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer_npoints3(test_data):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval..

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n_points= 3, last_full=False
    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
    ref_references = [np.median(inter) for inter in ref_intervals]
    ref_boundaries = [[1.2, 2.45], [2.45, 3.3], [3.3, 4.5], [4.5, 5.0]]
    n_points_slicer = PointsPerIntervalSlicer(3, last_full=False, min_n_points=1)
    my_slices, my_references, my_boundaries = n_points_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer_npoints3_lastfull_minpoints3(test_data):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval..

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n_points= 3, last_full=True, min_n_points=3
    ref_intervals = [[1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
    ref_references = [np.median(inter) for inter in ref_intervals]
    ref_boundaries = [[1.5, 2.55], [2.55, 3.55], [3.55, 5.0]]
    n_points_slicer = PointsPerIntervalSlicer(3, last_full=True, min_n_points=3)
    my_slices, my_references, my_boundaries = n_points_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_points_per_interval_slicer_npoints3_lastfull_reference_mean(test_data):
    """
    Function to compare characterisics of reference interval generated by
    PointsPerIntervalSlicer and manually generated interval..

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n_points= 3, last_full=True, reference=np.mean
    ref_intervals = [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
    ref_references = [np.mean(inter) for inter in ref_intervals]
    ref_boundaries = [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]]
    n_points_slicer = PointsPerIntervalSlicer(3, reference=np.mean, min_n_points=1)
    my_slices, my_references, my_boundaries = n_points_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints2_include_max(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n=2 include_max=True
    ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]]
    ref_references = [2.15, 4.05]
    ref_boundaries = [[1.2, 3.1], [3.1, 5.0]]
    number_slicer = NumberOfIntervalsSlicer(2, min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints2(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    # n=2, include_max=False
    ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0]]
    ref_references = [2.15, 4.05]
    ref_boundaries = [[1.2, 3.1], [3.1, 5.0]]
    number_slicer = NumberOfIntervalsSlicer(2, include_max=False, min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints3_include_max(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 3 and the upper boundary of the last 
    interval is inclusive.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0, 5.0]]
    ref_width = (5 - 1.2) / 3
    ref_references = [
        1.2 + ref_width / 2,
        1.2 + 3 * ref_width / 2,
        1.2 + 5 * ref_width / 2,
    ]
    ref_boundaries = [(c - ref_width / 2, c + ref_width / 2) for c in ref_references]
    number_slicer = NumberOfIntervalsSlicer(3, min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_almost_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints3_numberintervals2_include_max(
    test_data,
):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 3, the upper boundary of the last 
    interval is inclusive, the minimum points per intervals = 3 and the
    minimum number of intervals= 2. 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]]
    ref_width = (5 - 1.2) / 3
    ref_references = [1.2 + ref_width / 2, 1.2 + 3 * ref_width / 2]
    ref_boundaries = [(c - ref_width / 2, c + ref_width / 2) for c in ref_references]
    number_slicer = NumberOfIntervalsSlicer(3, min_n_points=3, min_n_intervals=2)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_almost_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints3(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 3, the upper boundary of the last 
    interval is not inclusive.

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0]]
    ref_width = (5 - 1.2) / 3
    ref_references = [
        1.2 + ref_width / 2,
        1.2 + 3 * ref_width / 2,
        1.2 + 5 * ref_width / 2,
    ]
    ref_boundaries = [(c - ref_width / 2, c + ref_width / 2) for c in ref_references]
    number_slicer = NumberOfIntervalsSlicer(3, include_max=False, min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_almost_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints2_include_max_valuerange(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 2, the upper boundary of the last 
    interval is inclusive, and the the value range used for creating the
    intervals is 0-5. 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0]]
    ref_references = [1.25, 3.75]
    ref_boundaries = [[0, 2.5], [2.5, 5.0]]
    number_slicer = NumberOfIntervalsSlicer(2, value_range=(0, 5), min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints2_include_max_reference_median(
    test_data,
):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 2, the upper boundary of the last 
    interval is inclusive and the reference of the interval is the median.
    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]]
    ref_references = [2.4, 3.6]
    ref_boundaries = [[1.2, 3.1], [3.1, 5.0]]
    number_slicer = NumberOfIntervalsSlicer(2, reference=np.median, min_n_points=1)
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])


def test_number_of_interval_slicer_numberpoints3_numberintervals3(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the number of intervals= 3, the upper boundary of the last 
    interval is inclusive, the minimum points per intervals = 3 and the
    minimum number of intervals= 3. 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]]
    ref_width = (5 - 1.2) / 3
    ref_references = [1.2 + ref_width / 2, 1.2 + 3 * ref_width / 2]
    ref_boundaries = [(c - ref_width / 2, c + ref_width / 2) for c in ref_references]
    number_slicer = NumberOfIntervalsSlicer(3, min_n_points=3, min_n_intervals=3)
    try:
        my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
    except RuntimeError:
        pass  # we expect a RuntimeError if there are less than min_n_intervals intervals


def test_width_of_interval_slicer_reference_left_right_open(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the reference of the interval is the lower boundary (left) and the 
    upper interval boundary is open (right open). 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_references = [1, 2, 3, 4, 5]
    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]]
    ref_boundaries = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    width_slicer = WidthOfIntervalSlicer(width=1, reference="left", min_n_points=1)
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_width_of_interval_slicer_reference_left_left_open(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the reference of the interval is the lower boundary (left) and the 
    lower interval boundary is open (left open). 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_references = [1, 2, 3, 4]
    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]]
    ref_boundaries = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    width_slicer = WidthOfIntervalSlicer(
        width=1, reference="left", right_open=False, min_n_points=1
    )
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_width_of_interval_slicer_reference_center_right_open(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the reference of the interval is the center and the upper interval 
    boundary is open (right open). 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_references = [1.5, 2.5, 3.5, 4.5, 5.5]
    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]]
    ref_boundaries = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    width_slicer = WidthOfIntervalSlicer(width=1, min_n_points=1)
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_width_of_interval_slicer_reference_center_left_open(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval.  Specific test
    case: the reference of the interval is the center and the lower interval 
    boundary is open (left open). 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_references = [1.5, 2.5, 3.5, 4.5]
    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]]
    ref_boundaries = [[1, 2], [2, 3], [3, 4], [4, 5]]
    width_slicer = WidthOfIntervalSlicer(width=1, right_open=False, min_n_points=1)
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_width_of_interval_slicer_reference_median_right_open(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the reference of the interval is the median and the upper interval 
    boundary is open (right open). 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """

    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]]
    ref_references = [np.median(x) for x in ref_intervals]
    ref_boundaries = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    width_slicer = WidthOfIntervalSlicer(width=1, reference=np.median, min_n_points=1)
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])


def test_width_of_interval_slicer_reference_left_right_open_minpoints(test_data):
    """
    Function to compare characterisics of reference interval generated by
    NumberOfIntervalSlicer and manually generated interval. Specific test
    case: the reference of the interval is the lower boundary (left) and the 
    minimum points per interval=2. 

    Parameters
    ----------
    test_data : numpy array
        Data to test the interval slicer.
 
    Returns
    -------
    None.

    """
    # reference="left", right_open, min_n_points=2
    ref_references = [1, 2, 3]
    ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6]]
    ref_boundaries = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    width_slicer = WidthOfIntervalSlicer(width=1, reference="left", min_n_points=2)
    my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_references, ref_references)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
