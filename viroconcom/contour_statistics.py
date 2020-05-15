#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzes contours, for example sorts the coordintaes or checks whether a
datapoint is inside or outside a contour.
"""
import numpy as np
import matplotlib as mpl
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def points_outside(contour_x, contour_y, x, y):
    """
    Determines the observations outside the region enclosed by a 2D contour.

    Parameters
    ----------
    contour_x : ndarray of doubles
        The contour's coordinates in the x-direction.
    contour_y : ndarray of doubles
        The contour's coordiantes in the y-direction.
    x : ndarray of doubles
        The sample's first environmental variable.
    y : ndarray of doubles
        The sample's second environmental variable.
    Returns
    -------
    x_outside : nparray
        The observations that are outside of the contour of variable 1.
    y_outside : nparray
        The observations that are outside of the contour of variable 2.
    x_inside : nparray
        The observations that are inside of the contour of variable 1.
    y_inside : nparray
        The observations that are inside of the contour of variable 2.
    """
    contour_path = mpl.path.Path(np.column_stack((contour_x, contour_y)))
    is_inside_contour = contour_path.contains_points(np.column_stack((x, y)))
    indices_is_outside = np.argwhere(is_inside_contour == False)
    x_outside = x[indices_is_outside]
    y_outside = y[indices_is_outside]
    indices_is_inside = np.argwhere(is_inside_contour == True)
    x_inside = x[indices_is_inside]
    y_inside = y[indices_is_inside]

    return (x_outside, y_outside, x_inside, y_inside)


def sort_points_to_form_continous_line(x, y, do_search_for_optimal_start=False):
    """
    Sorts contour points to form a a continous line / contour.

    Thanks to https://stackoverflow.com/questions/37742358/
    sorting-points-to-form-a-continuous-line

    Parameters
    ----------
    x : array_like
    y : array_like
    Returns
    -------
    sorted_points : tuple of array_like floats
        The sorted points.
    """
    points = np.c_[x, y]
    clf = NearestNeighbors(2).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T, 0))

    xx = x[order]
    yy = y[order]

    if do_search_for_optimal_start:
        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
        mindist = np.inf
        minidx = 0

        for i in range(len(points)):
            p = paths[i]  # Order of nodes.
            ordered = points[p]  # Ordered nodes.
            # Find cost of that order by the sum of euclidean distances
            # between points (i) and (i + 1).
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i

        opt_order = paths[minidx]

        xx = x[opt_order]
        yy = y[opt_order]

    return (xx, yy)
