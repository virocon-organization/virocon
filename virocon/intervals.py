import numpy as np

from abc import ABC, abstractmethod


class IntervalSlicer(ABC):

    def __init__(self, **kwargs):
        self.min_n_points = kwargs.get("min_n_points", 50)
        self.min_n_intervals = kwargs.get("min_n_intervals", 3)
        self.center = None

    def slice_(self, data):
        interval_slices, interval_centers = self._slice(data)

        ok_slices, ok_centers = self._drop_too_small_intervals(interval_slices,
                                                               interval_centers)

        if len(interval_slices) < self.min_n_intervals:
            raise RuntimeError("Slicing resulting in too few intervals. "
                               f"Need at least {self.min_n_intervals}, "
                               f"but got only {len(interval_slices)} intervals.")

        if self.center is not None:
            # assert that center is a callable
            ok_centers = [self.center(data[slice_]) for slice_ in ok_slices]

        return ok_slices, ok_centers

    @abstractmethod
    def _slice(self, data):
        pass

    def _drop_too_small_intervals(self, interval_slices, interval_centers):
        ok_slices = []
        ok_centers = []
        for slice_, int_cent in zip(interval_slices, interval_centers):
            # slice_ is a bool array, so sum returns number of points in interval
            if np.sum(slice_) >= self.min_n_points:
                ok_slices.append(slice_)
                ok_centers.append(int_cent)
        return ok_slices, ok_centers


class WidthOfIntervalSlicer(IntervalSlicer):
    
    def __init__(self, width, center=None, offset=False, right_open=True, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.center = center
        self.offset = offset
        self.right_open = right_open
        
    def _slice(self, data):
        # TODO floor min with precision of width instead of 0
        data_min = 0
        data_max = np.max(data)
        width = self.width
        interval_centers = np.arange(data_min, data_max + width, width)
        if self.offset:
            interval_centers += 0.5 * width
            
        if self.right_open:
            interval_slices = [((int_cent - 0.5 * width <= data) & 
                                (data < int_cent + 0.5 * width))
                               for int_cent in interval_centers]
        else:
            interval_slices = [((int_cent - 0.5 * width < data) & 
                                (data <= int_cent + 0.5 * width))
                               for int_cent in interval_centers]

        return interval_slices, interval_centers
    
    
class NumberOfIntervalsSlicer(IntervalSlicer):
    
    def __init__(self, n_intervals, center=None, include_max=True, range_=None, **kwargs):
        super().__init__(**kwargs)
        if n_intervals < self.min_n_intervals:
            self.min_n_intervals = n_intervals
        self.n_intervals = n_intervals
        self.center = center
        self.include_max = include_max
        self.range_ = range_

    def _slice(self, data):
        if self.range_ is not None:
            range_ = self.range_ 
        else:
            range_ = (min(data), max(data))
        
        interval_starts, interval_width = np.linspace(range_[0], 
                                                      range_[1],
                                                      num=self.n_intervals,
                                                      endpoint=False,
                                                      retstep=True
                                                      )
        interval_centers = interval_starts + 0.5 * interval_width
        interval_slices = [((data >= int_start) & 
                            (data < int_start + interval_width)) 
                           for int_start in interval_starts[:-1]]
        
        # include max in last interval ?
        int_start = interval_starts[-1]
        if self.include_max:
            interval_slices.append(((data >= int_start) & (data <= int_start + interval_width)))
        else:
            interval_slices.append(((data >= int_start) & (data < int_start + interval_width)))
            
        return interval_slices, interval_centers


class PointsPerIntervalSlicer(IntervalSlicer):

    def __init__(self, n_points, center=None, last_full=True, **kwargs):
        super().__init__(**kwargs)
        if n_points < self.min_n_points:
            self.min_n_points = n_points
            
        self.n_points = n_points
        self.center = center if center is not None else np.median
        self.last_full = last_full

    def _slice(self, data):
        sorted_idc = np.argsort(data) 
        n_full_chunks = len(data) // self.n_points 
        remainder = len(data) % self.n_points
        if remainder != 0:
            if self.last_full:
                interval_idc = np.split(sorted_idc[remainder:], 
                                        n_full_chunks)
                interval_idc.insert(0, sorted_idc[:remainder])
                
            else:
                interval_idc = np.split(sorted_idc[:len(data)-remainder], 
                                        n_full_chunks)
                interval_idc.append(sorted_idc[len(data)-remainder:])
        else:
            interval_idc = np.split(sorted_idc, n_full_chunks)
            
        interval_slices = [np.isin(sorted_idc, idc, assume_unique=True) for idc in interval_idc]
        interval_centers = [None] * len(interval_slices)  # gets overwritten in super().slice_ anyway
        
        return interval_slices, interval_centers
