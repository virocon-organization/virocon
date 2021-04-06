import pytest

import numpy as np

from virocon import (WidthOfIntervalSlicer, NumberOfIntervalsSlicer, 
                     PointsPerIntervalSlicer)
@pytest.fixture(scope="module")
def test_data():
    return np.array([1.2, 1.5, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0 ])


@pytest.fixture(params=[
    ({"width" : 1, "min_n_points" : 1},
     [1, 2, 3, 4, 5],
     [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]],
     [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]),
    ({"width": 1, "right_open": False, "min_n_points": 1},
     [1, 2, 3, 4, 5],
     [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0], [5.0]],
     [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]),
    ({"width" : 1, "offset": True, "min_n_points" : 1},
     [1.5, 2.5, 3.5, 4.5, 5.5],
     [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]],
     [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]),
    ({"width" : 1, "offset": True, "right_open": False, "min_n_points" : 1},
     [1.5, 2.5, 3.5, 4.5],
     [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]],
     [[1, 2], [2, 3], [3, 4], [4, 5]]),
    ({"width" : 1, "center": np.median, "min_n_points" : 1},
     [1.2, 1.95, 2.6, 3.6, 5.0],
     [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]],
     [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]),
    ({"width" : 1, "min_n_points" : 2},
     [2, 3, 4],
     [[1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0]],
     [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]),
    ])
def woi_params_and_ref(request):
    params = request.param[0]
    centers = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {"params" : params,
            "centers" : centers,
            "intervals" : intervals,
            "boundaries" : boundaries}

@pytest.fixture(params=[
    ({"n_intervals" : 2, "min_n_points" : 1, "min_n_intervals" : 2},
     [2.15, 4.05],
     [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]],
     [[1.2, 3.1], [3.1, 5.0]]),
    ({"n_intervals" : 2, "include_max": False, "min_n_points" : 1,},
     [2.15, 4.05],
     [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0]],
     [[1.2, 3.1], [3.1, 5.0]]),
    ({"n_intervals" : 3, "min_n_points" : 1},
     [1.8333333333333333, 3.0999999999999996, 4.366666666666666],
     [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0, 5.0]],
     [[1.2, 2.466666666666667], [2.4666666666666663, 3.733333333333333],[3.733333333333333, 5.0]]),
    ({"n_intervals" : 3, "min_n_points" : 3, "min_n_intervals" : 2},
     [1.8333333333333333, 3.0999999999999996],
     [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]],
     [[1.2, 2.466666666666667], [2.4666666666666663, 3.733333333333333]]),
    ({"n_intervals" : 3,"include_max": False, "min_n_points" : 1},
     [1.8333333333333333, 3.0999999999999996, 4.366666666666666],
     [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0]],
     [[1.2, 2.466666666666667], [2.4666666666666663, 3.733333333333333], [3.733333333333333, 5.0]]),
    ({"n_intervals" : 2, "range_": (0,5), "min_n_points" : 1,},
     [1.25, 3.75],
     [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0]],
     [[0, 2.5], [2.5, 5.0]]),
    ({"n_intervals" : 2, "center": np.median, "min_n_points" : 1,},
     [2.4, 3.6],
     [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]],
     [[1.2, 3.1], [3.1, 5.0]]),
    ({"n_intervals" : 3, "min_n_points" : 3}, # should raise RuntimeError (too few intervals)
     None,
     None,
     None),
    ])
def noi_params_and_ref(request):
    params = request.param[0]
    centers = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {"params" : params,
            "centers" : centers,
            "intervals" : intervals,
            "boundaries" : boundaries}


@pytest.fixture(params=[
    ({"n_points" : 2, "min_n_points" : 1},
     [1.35, 2.45, 2.85, 3.55, 4.5],
     [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1], [3.5, 3.6], [4.0, 5.0]],
     [[1.2, 1.95], [1.95, 2.55], [2.55, 3.3], [3.3, 3.8], [3.8, 5.0]]),
    ({"n_points" : 3, "min_n_points" : 1},
     [1.2, 2.4, 3.1, 4.0],
     [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
     [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]]),
    ({"n_points" : 3, "last_full": False, "min_n_points" : 1},
     [1.5, 2.6, 3.6, 5.0],
     [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]],
     [[1.2, 2.45], [2.45, 3.3], [3.3, 4.5], [4.5, 5.0]]),
    ({"n_points" : 3,},
     [2.4, 3.1, 4.0],
     [[1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
     [[1.5, 2.55], [2.55, 3.55], [3.55, 5.0]]),
    ({"n_points" : 3, "center": np.mean, "min_n_points" : 1},
     [1.2, 2.1333333333333333, 3.0666666666666664, 4.2],
     [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]],
     [[1.2, 1.35], [1.35, 2.55], [2.55, 3.55], [3.55, 5.0]]),
    ])
def ppi_params_and_ref(request):
    params = request.param[0]
    centers = request.param[1]
    intervals = request.param[2]
    boundaries = request.param[3]
    return {"params" : params,
            "centers" : centers,
            "intervals" : intervals,
            "boundaries" : boundaries}



def test_width_of_interval_slicer(test_data, woi_params_and_ref):
    
    params = woi_params_and_ref["params"]
    ref_centers = woi_params_and_ref["centers"]
    ref_intervals = woi_params_and_ref["intervals"]
    ref_boundaries = woi_params_and_ref["boundaries"]
    slicer = WidthOfIntervalSlicer(**params)
    
    my_slices, my_centers, my_boundaries = slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_centers, ref_centers)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
        
        
def test_number_of_intervals_slicer(test_data, noi_params_and_ref):
    
    params = noi_params_and_ref["params"]
    ref_centers = noi_params_and_ref["centers"]
    ref_intervals = noi_params_and_ref["intervals"]
    ref_boundaries = noi_params_and_ref["boundaries"]
    slicer = NumberOfIntervalsSlicer(**params)
    
    if ref_centers is None:
        with pytest.raises(RuntimeError):
            my_slices, my_centers, my_boundaries = slicer.slice_(test_data)
    else:
        my_slices, my_centers, my_boundaries = slicer.slice_(test_data)
        my_intervals = [test_data[slice_] for slice_ in my_slices]
        np.testing.assert_almost_equal(my_centers, ref_centers)
        assert len(my_intervals) == len(ref_intervals)
        for i in range(len(my_intervals)):
            np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
        for i in range(len(my_boundaries)):
            np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])
        
        
def test_points_per_interval_slicer(test_data, ppi_params_and_ref):
    
    params = ppi_params_and_ref["params"]
    ref_centers = ppi_params_and_ref["centers"]
    ref_intervals = ppi_params_and_ref["intervals"]
    ref_boundaries = ppi_params_and_ref["boundaries"]
    slicer = PointsPerIntervalSlicer(**params)
    
    my_slices, my_centers, my_boundaries = slicer.slice_(test_data)
    my_intervals = [test_data[slice_] for slice_ in my_slices]
    np.testing.assert_array_equal(my_centers, ref_centers)
    assert len(my_intervals) == len(ref_intervals)
    for i in range(len(my_intervals)):
        np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    for i in range(len(my_boundaries)):
        np.testing.assert_almost_equal(my_boundaries[i], ref_boundaries[i])
    




