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
        
        # read default values from function or set default as 1 if not specified
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
        
          
        

# class DependentDependencyFunction():
    
#     def __init__(self, func, bounds=None, constraints=None, **kwargs):
#         #TODO add fitting method 
#         self.func = func
#         self.bounds = bounds
#         self.constraints = constraints
        
#         # read default values from function or set default as 1 if not specified
#         sig = signature(func)
#         self.parameters = {par.name : (par.default if par.default is not par.empty else 1) 
#                            for par in list(sig.parameters.values())[1:]
#                            }
#         self._may_fit = True
#         self.dependent_parameters = {}
#         for key in kwargs.keys():
#             if key in self.parameters.keys():
#                 self._may_fit = False
#                 dep_param = kwargs[key]
#                 self.dependent_parameters[key] = dep_param
#                 dep_param.register(self)
#                 self.func = partial(self.func, key=dep_param)
#                 del self.parameters[key]
                
        
        
    
#     def __call__(self, x, *args, **kwargs):
#         if len(args) + len(kwargs) == 0:
#             return self.func(x, *self.parameters.values())
#         elif len(args) + len(kwargs) == len(self.parameters):
#             return self.func(x, *args, **kwargs)
#         else:
#             raise ValueError() # TODO helpful error message
        
#     def fit(self, x, y):
#         # save x and y, this also marks that we want to fit
#         self.x = x
#         self.y = y
#         if self._may_fit:  # is the conditioner fitted, so that we can fit now?
#             self._fit(self.x, self.y)
  
#     def _fit(self, x, y):
#         method = "lsq"
#         # alternative: "wlsq" for weighted least squares
#         bounds = self.bounds
#         constraints = self.constraints
                
#         # get initial parameters
#         p0 = tuple(self.parameters.values())
        
#         if constraints is None:
#             popt = fit_function(self, x, y, p0, method, bounds)
#         else:
#             popt = fit_constrained_function(self, x, y, p0, method, bounds, constraints)
        
#         # update self with fitted parameters
#         self.parameters = dict(zip(self.parameters.keys(), popt))
        
        
        
#     def callback(self, caller):
#         self.counter +=1
#         assert caller in self.dependent_parameters.values()
#         if self.counter >= len(self.dependent_parameters):
#             self._may_fit = True
#             if self.x and self.y: # did we try to fit earlier, but could not?
#                 self.fit(self.x, self.y)
        
        
        
# class HierarchicalDependenceFunction():
#     def __init__(self, dep_func_descriptions):
#         self.dep_func_descriptions = dep_func_descriptions
        
#         self.dep_funcs = []
#         self.conditional_on = [] #TODO check that conditionals are hierarchical, i.e. sorted ascending
#         self.conditional_parameters = []
#         for dep_func_desc in dep_func_descriptions:
#             self.dep_funcs.append(dep_func_desc["dependency_function"])
#             if "conditional_on" in dep_func_desc.keys():
#                 self.conditional_on.append(dep_func_desc["conditional_on"])
#                 self.conditional_parameters.append(dep_func_desc["conditional_parameters"])
#             else:
#                 self.conditional_on.append(None)
#                 self.conditional_parameters.append(None)
        
#         self.parameters = []
#         self.n_parameters = []
#         for dep_func in self.dep_funcs:
#             self.parameters.append(dep_func.parameters)
#             self.n_parameters.append(len(dep_func.parameters))
        
        
        
        
#     def __call__(self, x, *args):
#         if len(args)  == np.sum(self.n_parameters):
#             it = iter(args)
#             args_chunks = [[next(it) for _ in range(n)] 
#                            for n in self.n_parameters]
#             result = [None] * len(self.dep_funcs)
#             for i in range(len(self.dep_funcs)):
#                 result[i] = self.dep_funcs[i](x, *args_chunks[i])
                
#             return tuple(result)
                
#         else:
#             raise ValueError()
#                # TODO helpful error message
#             # TODO call both, sort para
             
            
            
#     def fit(self, x, ys):
#         for i in range(len(self.dep_funcs)):
#             y = ys[i]
#             dep_func = self.dep_funcs[i]
#             method = "lsq"
#             bounds = dep_func.bounds
#             constraints = dep_func.constraints
                    
#             # get initial parameters
#             if self.conditional_on[i] is None:
#                 p0 = tuple(dep_func.parameters.values())
#                 fit_func = dep_func
#             else: # freeze conditional params
#                 param_names = dep_func.parameters.keys()
#                 frozen_params = {}
#                 for par_name in self.conditional_parameters[i]:
#                     par_val = self.dep_funcs[i](x)
#                     frozen_params[par_name] = par_val
#                     param_names.remove(par_name)
                    
#                 # freeze parameters that are dependent
#                 fit_func = partial(dep_func, **frozen_params)
#                 # give an initial for the remaining parameters
#                 p0 = tuple((dep_func.parameters[name] for name in param_names))
                
                    
#             if constraints is None:
#                 popt = fit_function(fit_func, x, y, p0, method, bounds)
#             else:
#                 popt = fit_constrained_function(fit_func, x, y, p0, method, bounds, constraints)
            
#             # update self with fitted parameters
#             if self.conditional_on[i] is None:
#                 self.dep_funcs[i] = dict(zip(self.parameters.keys(), popt))
#             else:
#                 params = dict(zip(self.parameters.keys(), popt))
#                 params.update(frozen_params)
#                 self.dep_funcs[i] = params
