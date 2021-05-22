import re

from inspect import signature, getsourcelines

from functools import partial

import numpy as np

from virocon._fitting import fit_function, fit_constrained_function

__all__ = ["DependenceFunction"]

# TODO test that order of execution does not matter
# it should not matter if the dependent or the conditioner are fitted first
class DependenceFunction:
    """
    Function to describe the dependencies between the variables.
    
    The dependence function is a function for the parameters of the dependent
    variable.
    
    Parameters
    ----------
    func : callable
        Dependence functions for the parameter.
        Maps a conditioning value x and an arbitrary number of parameters to 
        the value of a distributions parameter y. 
        func(x, *args) -> y     
    bounds : list
        Boundaries for parameters of func.
        Fixed scalar boundaries for func's parameters.
        E.g. 0 <= z <= 0 . 
    constraints : dict
        More complex contraints modeled as unequality constraints with 
        functions of the parameters of func z. 
        I.e. c_j(z) >= 0 .
        For further explanation see:
        https://docs.scipy.org/doc/scipy-1.6.2/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
    weights : callable, optional
        If given, weighted least squares fitting instead of least squares is
        used. Defaults to None.
        Given the data as observation tuples (x_i, y_i) maps from the vector 
        x and y to the vector of weights. 
        E.g. lambda x, y : y to linearly weight the observations with y_i.
   
    Examples 
    --------
    The dependence function is a function for the parameters of the dependent
    variable. E.g.the zero-upcrossing period is dependent on the 
    significant wave height (Hs|Tp). Assuming, the zero-upcrossing period is
    lognormally distributed, the parameters mu and sigma are described as
    functions of the significant wave height (equations given by
    Haselsteiner et. al(2021) [1]_ ). 
    
    :math:`\\mu_{tz}(h_s) =  ln \\left(c_1 + c_2 \\sqrt{ \\frac{h_s)}{ g}} \\right)`
    
    :math:`\\sigma_{tz}(h_s) = c_3 + \\frac{c_4)}{1+ c_5h_s}` 
    
    References
    ----------
    .. [1] Andreas F. Haselsteiner, Ryan G. Coe, Lance Manuel, Wei Chai, 
        Bernt Leira, Guilherme Clarindo, C. Guedes Soares, Ásta Hannesdóttir,
        Nikolay Dimitrov, Aljoscha Sander, Jan-Hendrik Ohlendorf, 
        Klaus-Dieter Thoben, Guillaume de Hauteclocque, Ed Mackay, Philip Jonathan,
        Chi Qiao, Andrew Myers, Anna Rode, Arndt Hildebrandt, Boso Schmidt, 
        Erik Vanem, Arne Bang Huseby(2021)    
        A benchmarking exercise for environmental contours.
    
    """

    # TODO implement check of bounds and constraints
    def __init__(self, func, bounds=None, constraints=None, weights=None, **kwargs):
        # TODO add fitting method
        self.func = func
        self.bounds = bounds
        self.constraints = constraints
        self.weights = weights

        # Read default values from function or set default as 1 if not specified.
        sig = signature(func)
        self.parameters = {
            par.name: (par.default if par.default is not par.empty else 1)
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
                dep_param_dict = {key: dep_param}
                self.func = partial(self.func, **dep_param_dict)
                del self.parameters[key]

    def __call__(self, x, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            return self.func(x, *self.parameters.values())
        elif len(args) + len(kwargs) == len(self.parameters):
            return self.func(x, *args, **kwargs)
        else:
            raise ValueError()  # TODO helpful error message

    def __repr__(self):
        if isinstance(self.func, partial):
            func = self.func.func
        else:
            func = self.func
        params = ", ".join(
            [
                f"{par_name}={par_value}"
                for par_name, par_value in self.parameters.items()
            ]
        )
        dep_params = ", ".join(
            [
                f"{par_name}={par_value}"
                for par_name, par_value in self.dependent_parameters.items()
            ]
        )
        combined_params = params + ", " + dep_params
        combined_params = combined_params.strip(", ")
        return f"DependenceFunction(func={func.__name__}, {combined_params})"

    def fit(self, x, y):
        """
        Determine the parameters of the dependence function.
        
        Parameters
        ----------
        
        x : array-like
            Input data (data consists of n observations (x_i, y_i)).
        y :array-like
            Target data (data consists of n observations (x_i, y_i)).
            
        Raises
        ------
        RuntimeError
            if the fit fails.

        """
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
            method = "wlsq"  # weighted least squares
            weights = weights(x, y)  # raises TypeError if not a callable
        else:
            method = "lsq"  # least squares

        bounds = self.bounds
        constraints = self.constraints

        # get initial parameters
        p0 = tuple(self.parameters.values())

        if constraints is None:
            try:
                popt = fit_function(self, x, y, p0, method, bounds, weights)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to fit dependence function {self}."
                    "Consider choosing different bounds."
                ) from e
        else:
            # TODO proper error handling for constrained fit
            popt = fit_constrained_function(
                self, x, y, p0, method, bounds, constraints, weights
            )

        # update self with fitted parameters
        self.parameters = dict(zip(self.parameters.keys(), popt))

        # after fitting inform dependents:
        for dependent in self.dependents:
            dependent.callback(self)

    def register(self, dependent):
        """
        Register a dependent DependenceFunction.
        The callback method of all registered dependents is called once this
        DependenceFunction was fitted.
        
        Parameters
        ----------
        dependent : DependenceFunction
            The DependenceFunctions to register.
        """

        self.dependents.append(dependent)

    def callback(self, caller):
        """
        Call to signal, that caller was fitted.
        
        Parameters
        ----------
        caller : DependeneFunction
            The DependenceFunctiom that is now fitted.
        """

        assert caller in self.dependent_parameters.values()
        # TODO raise proper error otherwise
        self._fitted_conditioners.add(caller)
        # check that all conditioners are already fitted, then we may fit self
        if self._fitted_conditioners.issubset(self.dependent_parameters.values()):
            self._may_fit = True
            if hasattr(self, "x") and hasattr(
                self, "y"
            ):  # did we try to fit earlier, but could not?
                self.fit(self.x, self.y)
