import numpy as np

from scipy.optimize import minimize


def get_least_squares_error_func(func, x, y):
    
    def least_squares_error_func(p):
        return np.sum((func(x, *p) - y)**2)
    
    return least_squares_error_func


def fit_function(func, x, y, p0, method, bounds, constraints):
    
    if method == "lsq":
        error_func = get_least_squares_error_func(func, x, y)
    else:
        raise NotImplementedError("At this time only least squares (lsq) " 
                                  "fitting is supported.")
    
    if constraints is None:
        constraints = ()
        
    result = minimize(error_func, p0, 
                      method="SLSQP", 
                      constraints=constraints,
                      bounds=bounds,
                      options={'ftol': 1e-9, "disp":False},
                      )
    if not result.success:
        raise RuntimeError("Error during fitting in scipy.optimize.minimize. "
                           f"Error message was: \n {result.message}.")
    popt = result.x
    return popt