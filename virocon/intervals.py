"""
Interval definitions for the subsequent model fitting.
"""

import numpy as np

from abc import ABC, abstractmethod

__all__ = [
    "WidthOfIntervalSlicer",
    "NumberOfIntervalsSlicer",
    "PointsPerIntervalSlicer",
]


class IntervalSlicer(ABC):
    """
    Abstract base class for IntervalSlicer
        
    Sorts the conditional variable (e.g Tp|Hs) into intervals of the
    independent variable (Hs). 
        
    """

    def __init__(self, **kwargs):
        # check if there are unknown kwargs
        kwarg_keys = kwargs.keys()
        unknown_kwarg_keys = set(kwarg_keys).difference(
            {"min_n_intervals", "min_n_points"}
        )
        if len(unknown_kwarg_keys) != 0:
            raise TypeError(
                "__init__() got an unexpected keyword argument "
                f"'{unknown_kwarg_keys.pop()}'"
            )

        self.min_n_points = kwargs.get("min_n_points", 50)
        self.min_n_intervals = kwargs.get("min_n_intervals", 3)
        self.reference = None

    def slice_(self, data):
        """
        Slices the data into intervals of equal width.
        
        Parameters
        ----------   
        data : one-dimensional ndarray.
            Contains the data of the independent variable.
            
        Returns
        -------
        interval_slices: list of ndarray
            Boolean arrays with same length as data. One for each interval. 
            True where a value in data falls in the corresponding interval.

        interval_references: ndarray
            Reference points of intervals. Length equal to number of intervals.

        interval_boundaries: list of tuple
            List of (upper, lower) limit tuples. One tuple for each interval.
        
        """

        interval_slices, interval_references, interval_boundaries = self._slice(data)

        if len(interval_slices) < self.min_n_intervals:
            raise RuntimeError(
                "Slicing resulting in too few intervals. "
                f"Need at least {self.min_n_intervals}, "
                f"but got only {len(interval_slices)} intervals."
            )

        if callable(self.reference):
            interval_references = [
                self.reference(data[slice_]) for slice_ in interval_slices
            ]

        return interval_slices, interval_references, interval_boundaries

    @abstractmethod
    def _slice(self, data):
        pass

    def _drop_too_small_intervals(self, interval_slices, interval_references, interval_boundaries):
        ok_slices = []
        ok_references = []
        ok_boundaries= []
        for slice_, int_cent, int_bounds in zip(interval_slices, interval_references, interval_boundaries):
            # slice_ is a boolean array, so sum returns number of points in interval
            if np.sum(slice_) >= self.min_n_points:
                ok_slices.append(slice_)
                ok_references.append(int_cent)
                ok_boundaries.append(int_bounds)
        return ok_slices, ok_references, ok_boundaries


class WidthOfIntervalSlicer(IntervalSlicer):
    """
        IntervalSlicer that uses width of intervals to define intervals.
        
        Parameters
        ----------   
        width : float
            The width of each interval.
        reference : str or callable, optional
            Determines the reference value for each interval. 
            If a string the following keywords are available: 
            'center': use the center / midpoint of the interval as reference,
            'left': use the left / lower bound of the interval and 
            'right': use the right / upper bound of the interval as reference.
            If a callable, a function is expected, that maps from an array with 
            the values of an interval to the reference of that interval
            (e.g. np.median). Defaults to 'center'.
        right_open : boolean, optional
            Determines how the boundaries of the intervals are defined. Either 
            the left or the right boundary is inclusive. Defaults to True, 
            meaning the left boundary is inclusive and the right exclusive, 
            i.e. :math:`[a, b)`.
        value_range : tuple, optional
            Determines the value range used for creating the intervals. 
            If None, 0 and np.max(data) are used.
            If a 2-tuple it contains the lower and upper limit of the range. 
            If either entry of the tuple is None the default for that entry is 
            used. Defaults to None.
        min_n_points : int, optional
            Minimal number of points per interval. Intervals with fewer points 
            are discarded. Defaults to 50.
        min_n_intervals : int, optional
            Minimal number of intervals. Raises a RuntimeError if slicing 
            resulted in fewer intervals. Defaults to 3.
            
        Raises
        ------
        RuntimeError
            if slicing resulted in fewer than min_n_intervals intervals.
    """

    def __init__(
        self, width, reference="center", right_open=True, value_range=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.reference = reference
        self.right_open = right_open
        self.value_range = value_range

    def _slice(self, data):

        if self.value_range is None:
            data_min = 0
            data_max = np.max(data)
        else:
            if self.value_range[0] is not None:
                data_min = self.value_range[0]
            else:
                data_min = 0
            if self.value_range[1] is not None:
                data_max = self.value_range[1]
            else:
                data_max = np.max(data)

        width = self.width
        interval_references = np.arange(data_min, data_max + width, width)+ 0.5*width
        

        if self.right_open:
            interval_slices = [
                ((int_cent - 0.5 * width <= data) & (data < int_cent + 0.5 * width))
                for int_cent in interval_references
            ]
        else:
            interval_slices = [
                ((int_cent - 0.5 * width < data) & (data <= int_cent + 0.5 * width))
                for int_cent in interval_references
            ]
           
        interval_boundaries = [
            (c - width / 2, c + width / 2) for c in interval_references
        ]
            
        if isinstance(self.reference, str):
            if self.reference.lower() == "center":
                 pass  # interval_references are already center of intervals
            elif self.reference.lower() == "right":
                interval_references += 0.5* width
            elif self.reference.lower() == "left":
                interval_references -= 0.5* width
            else:
                raise ValueError(
                    "Unknown value for 'reference'. "
                    "Supported values are 'center', 'left', "
                    f"and 'right', but got '{self.reference}'."
                )
        elif callable(self.reference):
            pass  #  handled in super class
        else:
            raise TypeError(
                "Wrong type for reference. Expected str or callable, "
                f"but got {type(self.reference)}."
            ) 

        interval_slices, interval_references, interval_boundaries = self._drop_too_small_intervals(
            interval_slices, interval_references, interval_boundaries
        )


        return interval_slices, interval_references, interval_boundaries


class NumberOfIntervalsSlicer(IntervalSlicer):
    """     
        IntervalSlicer that uses a number of intervals to define intervals of 
        equal width.
        
        Parameters
        ----------   
        n_intervals : int
            Number of intervals the dataset is split into.
        reference : str or callable, optional
            Determines the reference value for each interval. 
            If a string the following keywords are available: 
            'center': use the center / midpoint of the interval as reference,
            'left': use the left / lower bound of the interval and 
            'right': use the right / upper bound of the interval as reference.
            If a callable, a function is expected, that maps from an array with 
            the values of an interval to the reference of that interval
            (e.g. np.median). Defaults to 'center'.
        include_max : boolean, optional
            Determines if the upper boundary of the last interval is inclusive.
            True if inclusive. Defaults to True.
        value_range : tuple or None, optional
            Determines the value range used for creating n_intervals equally 
            sized intervals. If a tuple it contains the upper and lower limit 
            of the range. If None the min and max of the data are used. 
            Defaults to None.
        min_n_points : int, optional
            Minimal number of points per interval. Intervals with fewer points 
            are discarded. Defaults to 50.
        min_n_intervals : int, optional
            Minimal number of intervals. Raises a RuntimeError if slicing 
            resulted in fewer intervals. Defaults to 3.
            
        Raises
        ------
        RuntimeError
            if slicing resulted in fewer than min_n_intervals intervals.
    """

    def __init__(
        self,
        n_intervals,
        reference="center",
        include_max=True,
        value_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_intervals < self.min_n_intervals:
            self.min_n_intervals = n_intervals
        self.n_intervals = n_intervals
        self.reference = reference
        self.include_max = include_max
        self.value_range = value_range

    def _slice(self, data):
        if self.value_range is not None:
            value_range = self.value_range
        else:
            value_range = (min(data), max(data))

        interval_starts, interval_width = np.linspace(
            value_range[0],
            value_range[1],
            num=self.n_intervals,
            endpoint=False,
            retstep=True,
        )
        interval_references = interval_starts + 0.5 * interval_width
        
        interval_boundaries = [
            (c - interval_width / 2, c + interval_width / 2)
            for c in interval_references
        ]
        
        if isinstance(self.reference, str):
            if self.reference.lower() == "center":
                pass  # default
            elif self.reference.lower() == "right":
                interval_references = interval_starts + interval_width
            elif self.reference.lower() == "left":
                interval_references = interval_starts
            else:
                raise ValueError(
                    "Unknown value for 'reference'. "
                    "Supported values are 'center', 'left', "
                    f"and 'right', but got '{self.reference}'."
                )
        elif callable(self.reference):
            pass  #  handled in super class
        else:
            raise TypeError(
                "Wrong type for reference. Expected str or callable, "
                f"but got {type(self.reference)}."
            )

        interval_slices = [
            ((data >= int_start) & (data < int_start + interval_width))
            for int_start in interval_starts[:-1]
        ]

        # include max in last interval ?
        int_start = interval_starts[-1]
        if self.include_max:
            interval_slices.append(
                ((data >= int_start) & (data <= int_start + interval_width))
            )
        else:
            interval_slices.append(
                ((data >= int_start) & (data < int_start + interval_width))
            )

        interval_slices, interval_references, interval_boundaries = self._drop_too_small_intervals(
            interval_slices, interval_references, interval_boundaries
        )


        return interval_slices, interval_references, interval_boundaries


class PointsPerIntervalSlicer(IntervalSlicer):
    """
        Uses a number of points per interval to define intervals.

        Sorts the data and splits it into intervals with the same number of 
        points. In general this results in intervals with varying width.
        
        Parameters
        ----------   
        n_points : int
            The number of points per interval.
        reference : callable, optional
            Determines the reference value for each interval. 
            A function is expected, that maps from an array with 
            the values of an interval to the reference of that interval. 
            Defaults to np.median.
        last_full : boolean, optional
            If it is not possible to split the data in chunks with the same 
            number of points, one interval will have fewer points. This 
            determines if the last or the first interval should have n_points 
            points. If True the last interval contains n_points points and the 
            first interval contains the remaining points. Defaults to True.
        min_n_points : int, optional
            Minimal number of points per interval. Intervals with fewer points 
            are discarded. Defaults to 50.
        min_n_intervals : int, optional
            Minimal number of intervals. Raises a RuntimeError if slicing 
            resulted in fewer intervals. Defaults to 3.
            
        Raises
        ------
        RuntimeError
            if slicing resulted in fewer than min_n_intervals intervals.
    """

    def __init__(self, n_points, reference=np.median, last_full=True, **kwargs):
        super().__init__(**kwargs)
        if n_points < self.min_n_points:
            self.min_n_points = n_points

        self.n_points = n_points
        self.reference = reference
        self.last_full = last_full

    def _slice(self, data):
        sorted_idc = np.argsort(data)
        n_full_chunks = len(data) // self.n_points
        remainder = len(data) % self.n_points
        if remainder != 0:
            if self.last_full:
                interval_idc = np.split(sorted_idc[remainder:], n_full_chunks)
                interval_idc.insert(0, sorted_idc[:remainder])

            else:
                interval_idc = np.split(
                    sorted_idc[: len(data) - remainder], n_full_chunks
                )
                interval_idc.append(sorted_idc[len(data) - remainder :])
        else:
            interval_idc = np.split(sorted_idc, n_full_chunks)

        interval_slices = [
            np.isin(sorted_idc, idc, assume_unique=True) for idc in interval_idc
        ]
        interval_references = [None] * len(
            interval_slices
        )  # gets overwritten in super().slice_ anyway
        
        # Pass interval_references twice instead of boundaries. We calculate
        # boundaries later.
        interval_slices, interval_references, _ = self._drop_too_small_intervals(
            interval_slices, interval_references, interval_references
        )

        # calculate the interval boundaries
        # the boundary between two intervals shall be the mean of
        # the max of the lower interval and the min of the higher interval
        # for the first interval the lower limit is the min of the data in that interval
        # for the last interval the upper limit is the max of the data in that interval
        interval_boundaries = []
        lower_boundary = np.min(data[interval_slices[0]])
        interval = data[interval_slices[0]]
        for i in range(len(interval_slices) - 1):
            # calculate boundaries for ith interval
            next_interval = data[interval_slices[i + 1]]
            upper_boundary = (np.max(interval) + np.min(next_interval)) / 2
            interval_boundaries.append((lower_boundary, upper_boundary))
            # prepare variables for next interval
            lower_boundary = upper_boundary
            interval = next_interval

        # append boundaries for last interval
        upper_boundary = np.max(interval)
        interval_boundaries.append((lower_boundary, upper_boundary))

        return interval_slices, interval_references, interval_boundaries
