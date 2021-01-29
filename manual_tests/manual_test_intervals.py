
import numpy as np

from virocon.intervals import (WidthOfIntervalSlicer, NumberOfIntervalsSlicer, 
                               PointsPerIntervalSlicer)


test_data = np.array([1.2, 1.5, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0 ])
# %% test PointsPerIntervalSlicer()

# n_points= 2
ref_intervals = [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1], [3.5, 3.6], [4.0, 5.0]]
ref_centers = [np.median(inter) for inter in ref_intervals]
n_points_slicer = PointsPerIntervalSlicer(2, min_n_points=1)
my_slices, my_centers = n_points_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n_points= 3, last_full=True
ref_intervals = [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
ref_centers = [np.median(inter) for inter in ref_intervals]
n_points_slicer = PointsPerIntervalSlicer(3, last_full=True, min_n_points=1)
my_slices, my_centers = n_points_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n_points= 3, last_full=False
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
ref_centers = [np.median(inter) for inter in ref_intervals]
n_points_slicer = PointsPerIntervalSlicer(3, last_full=False, min_n_points=1)
my_slices, my_centers = n_points_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# n_points= 3, last_full=True, min_n_points=3
ref_intervals = [[1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
ref_centers = [np.median(inter) for inter in ref_intervals]
n_points_slicer = PointsPerIntervalSlicer(3, last_full=True, min_n_points=3)
my_slices, my_centers = n_points_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# n_points= 3, last_full=True, center=np.mean
ref_intervals = [[1.2], [1.5, 2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0, 5.0]]
ref_centers = [np.mean(inter) for inter in ref_intervals]
n_points_slicer = PointsPerIntervalSlicer(3, center=np.mean, min_n_points=1)
my_slices, my_centers = n_points_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    

# %% test NumberOfIntervalsSlicer()

# n=2 include_max=True
ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]]
ref_centers = [2.15, 4.05]
number_slicer = NumberOfIntervalsSlicer(2, min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=2, include_max=False
ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0]]
ref_centers = [2.15, 4.05]
number_slicer = NumberOfIntervalsSlicer(2, include_max=False, min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=3 include_max=True
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0, 5.0]]
ref_width = (5-1.2) / 3
ref_centers = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2, 1.2 + 5 * ref_width/2]
number_slicer = NumberOfIntervalsSlicer(3, min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_almost_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=3 include_max=True, min_n_points=3
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6]]
ref_width = (5-1.2) / 3
ref_centers = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2]
number_slicer = NumberOfIntervalsSlicer(3, min_n_points=3)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_almost_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=3 include_max=False
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6], [4.0]]
ref_width = (5-1.2) / 3
ref_centers = [1.2 + ref_width/2, 1.2 + 3 * ref_width/2, 1.2 + 5 * ref_width/2]
number_slicer = NumberOfIntervalsSlicer(3, include_max=False, min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_almost_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=2 include_max=True, range_=(0, 5)
ref_intervals = [[1.2, 1.5, 2.4], [2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0]]
ref_centers = [1.25, 3.75]
number_slicer = NumberOfIntervalsSlicer(2, range_=(0,5), min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# n=2 include_max=True, center=np.median
ref_intervals = [[1.2, 1.5, 2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0, 5.0]]
ref_centers = [2.4, 3.6]
number_slicer = NumberOfIntervalsSlicer(2, center=np.median, min_n_points=1)
my_slices, my_centers = number_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])

# %% test WidthOfIntervalSlicer()



# center=mid, right_open, offset=False
ref_centers = [1, 2, 3, 4, 5]
ref_intervals = [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
width_slicer = WidthOfIntervalSlicer(width=1, min_n_points=1)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# center=mid, left_open, offset=False
ref_centers = [1, 2, 3, 4, 5]
ref_intervals = [[1.2, 1.5], [2.4, 2.5], [2.6, 3.1, 3.5], [3.6, 4.0], [5.0]]
width_slicer = WidthOfIntervalSlicer(width=1, right_open=False, min_n_points=1)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# center=mid, right_open, offset=True
ref_centers = [1.5, 2.5, 3.5, 4.5, 5.5]
ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6], [4.0], [5.0]]
width_slicer = WidthOfIntervalSlicer(width=1, offset=True, min_n_points=1)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
# center=mid, left_open, offset=True
ref_centers = [1.5, 2.5, 3.5, 4.5]
ref_intervals = [[1.2, 1.5], [2.4, 2.5, 2.6], [3.1, 3.5, 3.6, 4.0], [5.0]]
width_slicer = WidthOfIntervalSlicer(width=1, offset=True, right_open=False, min_n_points=1)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# center=median, right_open, offset=False
ref_intervals = [[1.2], [1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0], [5.0]]
ref_centers = [np.median(x) for x in ref_intervals]
width_slicer = WidthOfIntervalSlicer(width=1, center=np.median, min_n_points=1)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])
    
    
# center=mid, right_open, offset=False, min_n_points=2
ref_centers = [2, 3, 4]
ref_intervals = [[1.5, 2.4], [2.5, 2.6, 3.1], [3.5, 3.6, 4.0]]
width_slicer = WidthOfIntervalSlicer(width=1, min_n_points=2)
my_slices, my_centers = width_slicer.slice_(test_data)
my_intervals = [test_data[slice_] for slice_ in my_slices]
np.testing.assert_array_equal(my_centers, ref_centers)
assert len(my_intervals) == len(ref_intervals)
for i in range(len(my_intervals)):
    np.testing.assert_array_equal(my_intervals[i], ref_intervals[i])