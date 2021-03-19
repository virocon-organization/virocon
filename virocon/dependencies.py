from inspect import signature

from functools import partial

import numpy as np

from virocon.fitting import fit_function, fit_constrained_function

#TODO test that order of execution does not matter
# it should not matter if the dependent or the conditioner are fitted first
class DependenceFunction():
    
    #TODO implement check of bounds and constraints
    def __init__(self, func, bounds=None, constraints=None, weights=None, **kwargs):
        #TODO add fitting method 
        self.func = func
        self.bounds = bounds
        self.constraints = constraints
        self.weights = weights
        
        # Read default values from function or set default as 1 if not specified.
        sig = signature(func)
        self.parameters = {par.name : (par.default if par.default is not par.empty else 1) 
                           for par in list(sig.parameters.values())[1:]
                           }
        
        self.dependents = []
        
        self._may_fit = True
        self.dependent_parameters = {}
        self._fitted_conditioners = set()
        for key in kwargs.keys():
            if key in self.parameters.keys():
                self._may_fit = False
                dep_param = kwargs[key]
                self.dependent_parameters[key] = dep_param
                dep_param.register(self)
                dep_param_dict = {key : dep_param}
                self.func = partial(self.func, **dep_param_dict)
                del self.parameters[key]
        
        
    def __call__(self, x, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            return self.func(x, *self.parameters.values())
        elif len(args) + len(kwargs) == len(self.parameters):
            return self.func(x, *args, **kwargs)
        else:
            raise ValueError() # TODO helpful error message
            
            
    def fit(self, x, y):
        # The dependence function does not know in which order all the
        # dependence functions are fitted.
        # If another DependenceFunction has to be fitted before the current one,
        # the current one will not be fitted.
        # In the init the current dependence function registered at all
        # dependence functions which it depends on.
        # After fitting, every dependence functions signals all it's registered
        # dependence functions that it was fitted,
        # so that they know they may be fitted as well.

        # save x and y, this also marks that fit was called
        self.x = x
        self.y = y
        if self._may_fit:  # is the conditioner fitted, so that we can fit now?
            self._fit(self.x, self.y)
            
    def _fit(self, x, y):
        weights = self.weights
        if weights is not None:
            method = "wlsq" # weighted least squares
        else:
            method = "lsq"  # least squares
            
        bounds = self.bounds
        constraints = self.constraints
        
        if weights is not None:
            if isinstance(weights, str):
                if weights.lower() == "linear":
                    weights = x  / np.sum(x)
                elif weights.lower() == "quadratic":
                    weights = x ** 2 / np.sum(x ** 2)
                elif weights.lower() == "cubic":
                    weights = x ** 3 / np.sum(x ** 3)
                else:
                    raise ValueError(f"Unsupported value for weights={weights}.")
            elif callable(weights):
                weights = weights(x, y)
            else:
                try:
                    _ = iter(weights)
                    weights = np.asarray_chkfinite(weights)
                except TypeError:
                    raise ValueError(f"Unsupported value for weights={weights}.")
        # get initial parameters
        p0 = tuple(self.parameters.values())
        
        if constraints is None:
            popt = fit_function(self, x, y, p0, method, bounds, weights)
        else:
            popt = fit_constrained_function(self, x, y, p0, method, bounds, constraints, weights)
        
        # update self with fitted parameters
        self.parameters = dict(zip(self.parameters.keys(), popt))
        
        # after fitting inform dependents:
        for dependent in self.dependents:
            dependent.callback(self)
            
        
        
    def register(self, dependent):
        self.dependents.append(dependent)
        
        
    def callback(self, caller):
        assert caller in self.dependent_parameters.values()
        # TODO raise proper error otherwise
        self._fitted_conditioners.add(caller)
        # check that all conditioners are already fitted, then we may fit self
        if self._fitted_conditioners.issubset(self.dependent_parameters.values()):
            self._may_fit = True
            if hasattr(self, "x") and hasattr(self, "y"): # did we try to fit earlier, but could not?
                self.fit(self.x, self.y)

