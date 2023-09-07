"""
Various functions that do not fit in an existing module.
"""

import networkx as nx
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.neighbors import NearestNeighbors

from virocon._intersection import intersection
from typing import Union
import numpy.typing as npt

__all__ = [
    "read_ec_benchmark_dataset",
    "calculate_design_conditions",
    "sort_points_to_form_continuous_line",
]

ROOT_DIR = Path(__file__).parent.parent


def read_ec_benchmark_dataset(file_path=None):
    """
    Reads an EC benchmark dataset.

    Reads a text/csv file as it is used in the EC benchmark:
    https://github.com/ec-benchmark-organizers/ec-benchmark

    Parameters
    ----------
    file_path : str
        Path to the dataset file. Defaults to the example dataset A.

    Returns
    -------
    dataset : pandas.DataFrame
        The dataset stored in a pandas DataFrame.
        Use dataset.values to access the underlying numpy array.
    """
    if file_path is None:
        file_path = str(ROOT_DIR.joinpath(Path("datasets/ec-benchmark_dataset_A.txt")))

    data = pd.read_csv(file_path, sep=";", skipinitialspace=True)
    data.index = pd.to_datetime(data.pop(data.columns[0]), format="%Y-%m-%d-%H")
    return data


def calculate_design_conditions(contour, steps: Union[list, int] = None, swap_axis=False) -> npt.ArrayLike:
    """Calculates design conditions (relevant points for structural analysis) for a contour.

    For example, in a sea state contour, one is interested in the upper Hs part for
    a given Tz/Tp. And one often likes to have points that are evenly spaced in
    the Tz/Tp direction, e.g. one Hs-Tz design condition per 0.5 s of Tz.

    Args:
        contour (Contour): Environmental conntour object.
        steps (list of floats or int, optional): Either a list of design conditoin x-values or the number how many
            steps shall be used (then the x-values will be evenly spaced). Defaults to None.
        swap_axis (bool, optional): Wheter x and y shall be swapped. Defaults to False.

    Returns:
        npt.ArrayLike: design conditions (points for structural analysis calculations).
    """
    if swap_axis:
        x_idx = 1
        y_idx = 0
    else:
        x_idx = 0
        y_idx = 1

    coords = contour.coordinates

    x1 = np.append(coords[:, x_idx], coords[0, x_idx])
    y1 = np.append(coords[:, y_idx], coords[0, y_idx])

    # Define the x-positions where a design condition will be calculated
    # The y-values will be found by calculating the intersections between
    # vertical lines and the contour coordinates.
    small_spacer = 0.0001 * (np.max(x1) - np.min(x1)) # Required to ensure that conditions on limits are picked up.
    default_lower_limit = np.min(x1) + small_spacer
    default_uppper_limit = np.max(x1) - small_spacer
    if steps is None:
        steps = np.linspace(default_lower_limit, default_uppper_limit, endpoint=True, num=10)
    else:
        try:
            iter(steps)  # if steps is iterable use it
        except TypeError:  # if steps is not iterable assume it's an int
            steps = np.linspace(default_lower_limit, default_uppper_limit, endpoint=True, num=steps)

    # Vertical line ylimits.
    y2 = [np.min(y1) - np.max(y1) * 0.1, np.max(y1) + np.max(y1) * 0.1]

    frontier_x = []
    frontier_y = []

    for x2 in steps:
        x, y = intersection(x1, y1, [x2, x2], y2)
        assert len(x) <= 2
        assert len(y) <= 2

        if len(y) == 0:
            continue

        frontier_x.append(x2)
        frontier_y.append(np.max(y))

    design_conditions = np.c_[frontier_x, frontier_y]
    return design_conditions


def sort_points_to_form_continuous_line(x, y, search_for_optimal_start=False):
    """
    Sorts contour points to form a a continous line / contour.

    This function simply sorts 2 dimensional points. The points are given
    by x and y coordinates. This function is used to sort the coordinates
    of 2 dimensional contours.

    Thanks to https://stackoverflow.com/a/37744549

    Parameters
    ----------
    x : array_like
        x-coordinate of contour points.
    y : array_like
        y-coordinate of contour points.
    search_for_optimal_start : boolean, optional
        If True, the algorithm also searches for the ideal starting node, see the
        stackoverflow link for more info. Defaults to False.

    Returns
    -------
    sorted_points : tuple of array_like floats
        The sorted points.
    """

    points = np.c_[x, y]
    clf = NearestNeighbors(n_neighbors=2).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T, 0))

    xx = x[order]
    yy = y[order]

    if search_for_optimal_start:
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

    return xx, yy
