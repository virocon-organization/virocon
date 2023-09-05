"""
Model fitting (parameter estimation).
"""

import numpy as np

from scipy.optimize import minimize, basinhopping, curve_fit

# %% unconstrained fitting


def fit_function(func, x, y, p0, method, bounds, weights=None):
    if bounds is not None:
        bounds = convert_bounds_for_curve_fit(bounds)

    if method == "lsq":
        if bounds is None:
            popt, _ = curve_fit(func, x, y, p0)
        else:
            popt, _ = curve_fit(func, x, y, p0, bounds=bounds)
    elif method == "wlsq":
        if bounds is None:
            popt, _ = curve_fit(func, x, y, p0, sigma=weights)
        else:
            popt, _ = curve_fit(func, x, y, p0, sigma=weights, bounds=bounds)
    else:
        raise ValueError(
            "method must be either lsq for least squares or"
            "wlsq for weighted least squares"
        )
    return popt


def convert_bounds_for_curve_fit(bounds):
    lower_bounds = []
    upper_bounds = []
    for lower, upper in bounds:
        lower_bounds.append(lower if lower is not None else -np.inf)
        upper_bounds.append(upper if upper is not None else np.inf)
    return [lower_bounds, upper_bounds]


# %% constrained fitting


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
        l = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        u = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    return cons


def fit_constrained_function(func, x, y, p0, method, bounds, constraints, weights=None):
    # p0 = np.asarray(p0)
    if method == "lsq":
        error_func = get_least_squares_error_func(func, x, y)
        # cost_func
    else:
        # TODO implement WLSQ
        raise NotImplementedError(
            "At this time only least squares (lsq) " "fitting is supported."
        )

    if constraints is None:
        constraints = []

    # constraints = list(constraints)
    # if bounds is not None:
    #     constraints.extend(bounds_to_constraints(bounds))

    result = minimize(
        error_func,
        p0,
        method="SLSQP",
        # jac="cs",
        # constraints=constraints,
        bounds=bounds,
        # tol=1E-15
        options={"eps": 1e-15}
        # options={'ftol': 1e-15, },#"disp":True},
    )
    # result = minimize(error_func, p0,
    #                   method="trust-constr",
    #                   #jac="3-point",
    #                   #constraints=constraints,
    #                   bounds=bounds,
    #                   #tol=1E-11
    #                   options={"initial_tr_radius": 10, },#"disp":True},
    #                   )

    # minimizer_kwargs = {"method" : "SLSQP",
    #                     "constraints" : constraints,
    #                     "bounds" : bounds,
    #                     "options" : {'ftol': 1e-15, "maxiter" : 1000},#"disp":True},
    #                     #"options": {"eps" : 1E-2}
    #                     }
    # minimizer_kwargs = {"method" : "SLSQP",
    #                     "constraints" : constraints,
    #                     "bounds" : None,
    #                     #"options" : {'ftol': 1e-15, "maxiter" : 1000},#"disp":True},
    #                     #"options": {"eps" : 1E-2}
    #                     }
    # res = basinhopping(error_func, p0, minimizer_kwargs=minimizer_kwargs, seed=42, )#stepsize=2, T=100)
    # result = res.lowest_optimization_result
    if not result.success:
        raise RuntimeError(
            "Error during fitting in scipy.optimize.minimize. "
            f"Error message was: \n {result.message}."
        )
    popt = result.x
    return popt
