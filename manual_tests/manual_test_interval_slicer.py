#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:02:27 2021

@author: lenastroer
"""

import numpy as np

from virocon import (
    WidthOfIntervalSlicer,
    NumberOfIntervalsSlicer,
    PointsPerIntervalSlicer,
)


test_data = np.array([1.2, 1.5, 2.4, 2.5, 2.6, 3.1, 3.5, 3.6, 4.0, 5.0])

# %% test WidthOfIntervalSlicer()

width_slicer_left= WidthOfIntervalSlicer(width=1, reference='left', min_n_points= 1)
w_slices_left, w_references_left, w_boundaries_left= width_slicer_left.slice_(test_data)


width_slicer_center= WidthOfIntervalSlicer(width=1, reference='center', min_n_points= 1)
w_slices_center, w_references_center, w_boundaries_center= width_slicer_center.slice_(test_data)


width_slicer_right= WidthOfIntervalSlicer(width=1, reference='right', min_n_points= 1)
w_slices_right, w_references_right, w_boundaries_right= width_slicer_right.slice_(test_data)


# %% test NumberOfIntervalSlicer()

number_slicer_left= NumberOfIntervalsSlicer(n_intervals=5, reference='left', min_n_points= 1)
n_slices_left, n_references_left, n_boundaries_left= number_slicer_left.slice_(test_data)


number_slicer_center= NumberOfIntervalsSlicer(n_intervals=5, reference='center', min_n_points= 1)
n_slices_center, n_references_center, n_boundaries_center= number_slicer_center.slice_(test_data)

number_slicer_right= NumberOfIntervalsSlicer(n_intervals=5, reference='right', min_n_points= 1)
n_slices_right, n_references_right, n_boundaries_right= number_slicer_right.slice_(test_data)

# %% test PointsPerIntervalSlicer()

points_slicer_left= PointsPerIntervalSlicer(n_points=2, reference='left', min_n_points= 1)
p_slices_left, p_references_left, p_boundaries_left= points_slicer_left.slice_(test_data)


points_slicer_center= PointsPerIntervalSlicer(n_points=2, reference='center', min_n_points= 1)
p_slices_center, p_references_center, p_boundaries_center= points_slicer_center.slice_(test_data)

points_slicer_right= PointsPerIntervalSlicer(n_points=2, reference='right', min_n_points= 1)
p_slices_right, p_references_right, p_boundaries_right= points_slicer_right.slice_(test_data)

