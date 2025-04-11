"""
Model fitting (parameter estimation).
"""

import numpy as np

from scipy.optimize import minimize, curve_fit


def fit_function(func, x, y, p0, method, bounds, weights=None):
    if bounds is not None:
        bounds = convert_bounds_for_curve_fit(bounds)

    kwargs = {}
    if method == "wlsq":
        kwargs["sigma"] = weights
    elif method != "lsq":
        raise ValueError(
            "method must be either lsq for least squares or"
            "wlsq for weighted least squares"
        )
    if bounds is not None:
        kwargs["bounds"] = bounds
    popt, _ = curve_fit(func, x, y, p0, **kwargs)
    return popt


def convert_bounds_for_curve_fit(bounds):
    lower_bounds = []
    upper_bounds = []
    for lower, upper in bounds:
        lower_bounds.append(lower if lower is not None else -np.inf)
        upper_bounds.append(upper if upper is not None else np.inf)
    return [lower_bounds, upper_bounds]


def get_least_squares_error_func(func, x, y):
    def least_squares_error_func(p):
        return np.sum((func(x, *p) - y) ** 2)

    return least_squares_error_func


def bounds_to_constraints(bounds):
    #  https://stackoverflow.com/a/41761740
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)
    return cons


def fit_constrained_function(func, x, y, p0, method, bounds, constraints, weights=None):
    if method == "lsq":
        error_func = get_least_squares_error_func(func, x, y)
    else:
        # TODO implement WLSQ
        raise NotImplementedError(
            "At this time only least squares (lsq) fitting is supported."
        )

    if constraints is None:
        constraints = []

    result = minimize(
        error_func,
        p0,
        method="SLSQP",
        bounds=bounds,
        options={"eps": 1e-15},
    )
    if not result.success:
        raise RuntimeError(
            "Error during fitting in scipy.optimize.minimize. "
            f"Error message was: \n {result.message}."
        )
    popt = result.x
    return popt
