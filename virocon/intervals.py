import numpy as np

from abc import ABC, abstractmethod

# TODO check output: are there enough intervals?
# TODO reduce code duplicates
class IntervalSlicer(ABC):
    
    @abstractmethod
    def slice_(self, data):
        pass

class WidthOfIntervalSlicer(IntervalSlicer):
    
    def __init__(self, width, center=None, offset=False, right_open=True, min_number_of_points=50):
        self.width = width
        self.center = center
        self.offset = offset
        self.right_open = right_open
        self.min_number_of_points = min_number_of_points
        
    def slice_(self, data):
        #TODO floor min with precion of width instead of 0
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
            
        ok_slices = []
        ok_centers = []
        for slice_, int_cent in zip(interval_slices, interval_centers):
            # slice_ is a bool array, so sum returns number of points in interval
            if np.sum(slice_) >= self.min_number_of_points:
                ok_slices.append(slice_)
                ok_centers.append(int_cent)
                
        if self.center is not None:
            # assert that center is a callable
            ok_centers = [self.center(data[slice_]) for slice_ in ok_slices]
            
       
        return ok_slices, ok_centers
    
    
class NumberOfIntervalsSlicer(IntervalSlicer):
    
    def __init__(self, n_intervals, center=None, include_max=True, range_=None, min_number_of_points=50):
        self.n_intervals = n_intervals
        self.center = center
        self.include_max = include_max
        self.range_ = range_
        self.min_number_of_points = min_number_of_points
        
    def slice_(self, data):
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
            
        ok_slices = []
        ok_centers = []
        for slice_, int_cent in zip(interval_slices, interval_centers):
            # slice_ is a bool array, so sum returns number of points in interval
            if np.sum(slice_) >= self.min_number_of_points:
                ok_slices.append(slice_)
                ok_centers.append(int_cent)
                
        if self.center is not None:
            # assert that center is a callable
            ok_centers = [self.center(data[slice_]) for slice_ in ok_slices]
        
        return ok_slices, ok_centers
    
class PointsPerIntervalSlicer(IntervalSlicer):

    def __init__(self, n_points, center=None, last_full=True, min_n_points=50):
        if n_points < min_n_points:
            raise ValueError("n_points has to be >= min_n_points, but was "
                             f"n_points={n_points} < min_n_points={min_n_points}")
        self.n_points = n_points
        self.center = center if center is not None else np.median
        self.last_full = last_full
        self.min_n_points = min_n_points
        
        
        
    def slice_(self, data):
        #TODO floor min with precion of width instead of 0
        sorted_idc = np.argsort(data) 
        n_full_chunks = len(data) // self.n_points 
        remainder = len(data)%self.n_points
        if remainder != 0:
            if self.last_full:
                
                # interval_slices.extend(np.split(sorted_idc[remainder:], 
                #                                 n_full_chunks))
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
                
        if self.center is not None:
            # assert that center is a callable
            interval_centers = [self.center(data[slice_]) for slice_ in interval_slices]
        
        ok_slices = []
        ok_centers = []
        for slice_, int_cent in zip(interval_slices, interval_centers):
            # slice_ is a bool array, so sum returns number of points in interval
            if np.sum(slice_) >= self.min_n_points:
                ok_slices.append(slice_)
                ok_centers.append(int_cent)
                
            
        return ok_slices, ok_centers