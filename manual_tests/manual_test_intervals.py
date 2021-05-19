
import numpy as np

from virocon import (WidthOfIntervalSlicer, NumberOfIntervalsSlicer, 
                               PointsPerIntervalSlicer)


test_data = np.array([1.2, 1.5, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0 ])
# %% test PointsPerIntervalSlicer()

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
    
    

# %% test NumberOfIntervalsSlicer()

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
    
# n=3 include_max=True
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0, 5.0]]
ref_width = (5-1.2) / 3
ref_references = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2, 1.2 + 5 * ref_width/2]
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
    
# n=3 include_max=True, min_n_points=3, min_n_intervals=2
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]]
ref_width = (5-1.2) / 3
ref_references = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2]
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
    
# n=3 include_max=False
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0]]
ref_width = (5-1.2) / 3
ref_references = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2, 1.2 + 5 * ref_width/2]
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
    
# n=2 include_max=True, value_range=(0, 5)
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0]]
ref_references = [1.25, 3.75]
ref_boundaries = [[0, 2.5], [2.5, 5.0]]
number_slicer = NumberOfIntervalsSlicer(2, value_range=(0,5), min_n_points=1)
my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
    
# n=2 include_max=True, reference=np.median
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
    
    
# n=3 include_max=True, min_n_points=3, min_n_intervals=3
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]]
ref_width = (5-1.2) / 3
ref_references = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2]
ref_boundaries = [(c - ref_width / 2, c + ref_width / 2) for c in ref_references]
number_slicer = NumberOfIntervalsSlicer(3, min_n_points=3, min_n_intervals=3)
try:
    my_slices, my_references, my_boundaries = number_slicer.slice_(test_data)
except RuntimeError:
    pass # we expect a RuntimeError if there are less than min_n_intervals intervals

# %% test WidthOfIntervalSlicer()

# reference="left", right_open,
ref_references = [1, 2, 3, 4, 5]
ref_intervals = [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
ref_boundaries = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]
width_slicer = WidthOfIntervalSlicer(width=1, reference="left", min_n_points=1)
my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
    
    
# reference="left", left_open
ref_references = [1, 2, 3, 4, 5]
ref_intervals = [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0], [5.0]]
ref_boundaries = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]
width_slicer = WidthOfIntervalSlicer(width=1, reference="left", right_open=False, min_n_points=1)
my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
    
    
# reference="center", right_open
ref_references = [1.5, 2.5, 3.5, 4.5, 5.5]
ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]]
ref_boundaries = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
width_slicer = WidthOfIntervalSlicer(width=1, min_n_points=1)
my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
    
# reference="center", left_open
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
    
    
# reference=median, right_open
ref_intervals = [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
ref_references = [np.median(x) for x in ref_intervals]
ref_boundaries = [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]]
width_slicer = WidthOfIntervalSlicer(width=1, reference=np.median, min_n_points=1)
my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])
    
    
# reference="left", right_open, min_n_points=2
ref_references = [2, 3, 4]
ref_intervals = [[1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0]]
ref_boundaries = [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]
width_slicer = WidthOfIntervalSlicer(width=1, reference="left", min_n_points=2)
my_slices, my_references, my_boundaries = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_references, ref_references)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
for i in range(len(my_boundaries)):
    np.testing.assert_array_equal(my_boundaries[i], ref_boundaries[i])