import numpy as np
import matplotlib as mpl
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def points_outside(contour_x, contour_y, x, y):
    """
    Determines the observations outside the region enclosed by the contour.
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
    outside_x : nparray
        The observations that are outside of the contour of variable 1.
    outside_y : nparray
        The observations that are outside of the contour of variable 2.
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


def thetastar_to_theta(thetastar, xspread, yspread):
    """
    Parameters
    ----------
    thetastar : ndarray of floats
        Angle in the normalized coordinate system.
    xspread : float
        Spread of x (xmax - ymin).
    yspread : float
        Spread of y (ymax - amin).
    Returns
    -------
    theta : float,
        The angle theta in the original coordinate system. The angle is
        defined counter clockwise, 0 at (x=1, y=0) and is converted to be
        inside the interval [0 360).
    """
    theta = np.arctan2(np.sin(thetastar) * yspread, np.cos(thetastar) * xspread)
    for i, t in enumerate(theta):
        if t < 0:
            theta[i] = t + 2 * np.pi
    return theta


def sort_points_to_form_continous_line(x, y, do_search_for_optimal_start=False):
    """
    Sorts contour points to forma a continous line / contour.

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
            p = paths[i]  # order of nodes
            ordered = points[p]  # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i

        opt_order = paths[minidx]

        xx = x[opt_order]
        yy = y[opt_order]

    return (xx, yy)
