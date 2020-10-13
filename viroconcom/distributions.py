import math

import numpy as np
import scipy.stats as sts

from abc import ABC, abstractmethod
from inspect import signature


from viroconcom.fitting import fit_function

# The distributions parameters need to have an order, this order is given by
# the parameters dict. As of Python 3.7 dicts officially keep their order of creation.
# So this version is a requirement.
# (Though the dict order might work as well in 3.6)


class ConditionalDistribution():
    
    def __init__(self, distribution_class, parameters):
        # allow setting fitting initials on class creation?
        self.distribution_class = distribution_class
        self.param_names = distribution_class.parameters.keys()
        self.conditional_parameters = {}
        self.fixed_parameters = {}
        for par_name in self.param_names:
            if par_name not in parameters.keys():
                raise ValueError(f"Mandatory key {par_name} was not found in parameters.")
            
            if callable(parameters[par_name]):
                self.conditional_parameters[par_name] = parameters[par_name]
            else:
                self.fixed_parameters[par_name] = parameters[par_name]
        


        
    def _get_dist(self, given):
        
        unpacked_params = {}
        for par_name in self.param_names:
            if par_name in self.conditional_parameters.keys():
                unpacked_params[par_name] = self.conditional_parameters[par_name](given)
            else:
                unpacked_params[par_name] = self.fixed_parameters[par_name]
                        
        return self.distribution_class(**unpacked_params)
  
                
    def pdf(self, x, given):
        # possibly allow given as ndarray in the future?
        dist = self._get_dist(given)
        return dist.pdf(x)
    
    def cdf(self, x, given):
        dist = self._get_dist(given)
        return dist.cdf(x)
    
    def i_cdf(self, x, given):
        dist = self._get_dist(given)
        return dist.i_cdf(x)
        
    def draw_sample_(self, n, given):
        dist = self._get_dist(given)
        return dist.draw_sample(n)
    
    def fit(self, data, conditioning_values):
        self.distributions_per_interval = []
        self.parameters_per_interval = []
        self.conditioning_values = conditioning_values
        # fit distribution to each interval
        for interval_data in data:
            fixed = None
            if len(self.fixed_parameters) > 0:
                fixed = self.fixed_parameters
            dist = self.distribution_class.fit(interval_data, fixed=fixed)
            self.distributions_per_interval.append(dist)
            self.parameters_per_interval.append(dist.parameters)
            
        # fit dependency functions
        # TODO dependence statt dependency
        fitted_dependency_functions = {}
        for par_name, dep_func in self.conditional_parameters:
            fitted_dependency_functions[par_name] = self._fit_dependency_function(par_name, dep_func)
            
        self.conditional_parameters = fitted_dependency_functions
        

    def _fit_dependency_function(self, par_name, dep_func):

        method = getattr(dep_func, "fit_method", "lsq")
        # alternative: "wlsq" for weighted least squares
        bounds = getattr(dep_func, "bounds", None)
        constraints = getattr(dep_func, "constraints", None)
        
        # TODO conditioning values durchreichen anstatt zustand
        x = self.conditioning_values
        y = [params[par_name] for params in self.parameters_per_interval]
        
        # get initial parameters
        p0 = tuple(dep_func.parameters.values())
        
        popt = fit_function(dep_func, x, y, p0, method, bounds, constraints)
        
        # update the dependency function with fitted parameters
        dep_func.parameters = dict(zip(dep_func.parameters.keys(), popt))
        return dep_func
        
        
class DependencyFunction():
    
    #TODO implement check of bounds and constraints
    def __init__(self, func, bounds=None, constraints=None):
        #TODO add fitting method 
        self.func = func
        self.bounds = bounds
        self.constraints = constraints
        
        # read default values from function or set default as 1 if not specified
        sig = signature(func)
        self.parameters = {par.name : (par.default if par.default is not par.empty else 1) 
                           for par in list(sig.parameters.values())[1:]
                           }
        
        
    def __call__(self, x, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            return self.func(x, *self.parameters.values())
        elif len(args) + len(kwargs) == len(self.parameters):
            return self.func(x, *args, **kwargs)
        else:
            raise ValueError() # TODO helpful error message
            


        
            


class Distribution(ABC):
    """
    Abstract base class for distributions.
    """

    @abstractmethod
    def cdf(self, x,):
        """Cumulative distribution function."""

    @abstractmethod
    def pdf(self, x):
        """Probability density function."""

    @abstractmethod
    def i_cdf(self, prob):
        """Inverse cumulative distribution function."""
        
    @abstractmethod
    def fit(self, data, fixed=None):
        """Fit the distribution to the sampled data"""



class WeibullDistribution(Distribution):
    
    
    def __init__(self, lambda_=1, k=1, theta=0):
        
        self.lambda_ = lambda_  # scale
        self.k = k  # shape
        self.theta = theta  # loc
        
    @property
    def parameters(self):
        return {"lambda_" : self.lambda_,
                "k" : self.k,
                "theta" : self.theta}
        
    def cdf(self, x):
        return sts.weibull_min.cdf(x, c=self.k, scale=self.lambda_)

    def i_cdf(self, prob):
        return sts.weibull_min.ppf(prob, c=self.k, scale=self.lambda_)

    def pdf(self, x):
        return sts.weibull_min.pdf(x, c=self.k, scale=self.lambda_)
    
    def fit(self, samples, fixed=None):
        p0={"lambda_": self.lambda_, "k": self.k, "theta": self.theta}
        
        fparams = {}
        if fixed is not None:
            if "k" in fixed.keys():
                fparams["f0"] = fixed["k"]
            if "theta" in fixed.keys():
                fparams["floc"] = fixed["theta"]
            if "lambda_" in fixed.keys():
                fparams["fscale"] = fixed["lambda_"]
        
        self.k, self.theta, self.lambda_  = (
            sts.weibull_min.fit(samples, p0["k"], loc=p0["theta"], 
                                scale=p0["lambda_"], **fparams)
             )
        
        
class LogNormalDistribution(Distribution):
    
   
    def __init__(self, mu=0, sigma=1):
        
        self.mu = mu
        self.sigma = sigma  # shape
        self.scale = math.exp(mu)
        
    @property
    def parameters(self):
        return {"mu" : self.mu,
                "sigma" : self.sigma}
        
    def cdf(self, x):
        return sts.lognorm.cdf(x, s=self.sigma, scale=self.scale)

    def i_cdf(self, prob):
        return sts.lognorm.ppf(prob, s=self.sigma, scale=self.scale)

    def pdf(self, x):
        return sts.lognorm.pdf(x, s=self.sigma, scale=self.scale)
    
    def fit(self, samples, fixed=None):
        p0={"mu": self.mu, "sigma": self.sigma}
        
        fparams = {}
        if fixed is not None:
            if "sigma" in fixed.keys():
                fparams["f0"] = fixed["sigma"]
            if "mu" in fixed.keys():
                fparams["fscale"] = math.exp(fixed["mu"])
        
        scale0 = math.exp(p0["mu"])
        self.sigma, self.scale  = (
            sts.lognorm.fit(samples, p0["sigma"], scale=scale0, **fparams)
             )
        self.mu = math.log(self.scale)
        
        
    