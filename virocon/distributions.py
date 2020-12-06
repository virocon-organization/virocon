import math

import numpy as np
import scipy.stats as sts

from abc import ABC, abstractmethod
from scipy.optimize import fmin


from virocon.fitting import fit_function, fit_constrained_function

# The distributions parameters need to have an order, this order is defined by
# the parameters dict. As of Python 3.7 dicts officially keep their order of creation.
# So this version is a requirement.
# (Though the dict order might work as well in 3.6)

# TODO possibly use something like
# https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
class ConditionalDistribution():
    
    def __init__(self, distribution_class, parameters):
        # allow setting fitting initials on class creation?
        self.distribution_class = distribution_class
        self.param_names = distribution_class().parameters.keys()
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
    
    def icdf(self, x, given):
        dist = self._get_dist(given)
        return dist.icdf(x)
        
    def draw_sample_(self, n, given):
        dist = self._get_dist(given)
        return dist.draw_sample(n)
    
    def fit(self, data, conditioning_values):
        self.distributions_per_interval = []
        self.parameters_per_interval = []
        self.data_intervals = data
        self.conditioning_values = np.array(conditioning_values)
        # fit distribution to each interval
        for interval_data in data:
            fixed = None
            if len(self.fixed_parameters) > 0:
                fixed = self.fixed_parameters
            dist = self.distribution_class()
            dist.fit(interval_data, fixed=fixed)
            self.distributions_per_interval.append(dist)
            self.parameters_per_interval.append(dist.parameters)
            
        # fit dependence functions
        fitted_dependence_functions = {}
        for par_name, dep_func in self.conditional_parameters.items():
            fitted_dependence_functions[par_name] = self._fit_dependence_function(par_name, dep_func)
            
        self.conditional_parameters = fitted_dependence_functions
        

    def _fit_dependence_function(self, par_name, dep_func):

        method = getattr(dep_func, "fit_method", "lsq")
        # alternative: "wlsq" for weighted least squares
        bounds = getattr(dep_func, "bounds", None)
        constraints = getattr(dep_func, "constraints", None)
        
        # TODO conditioning values durchreichen anstatt zustand
        x = self.conditioning_values
        y = [params[par_name] for params in self.parameters_per_interval]
        
        # get initial parameters
        p0 = tuple(dep_func.parameters.values())
        
        if constraints is None:
            popt = fit_function(dep_func, x, y, p0, method, bounds)
        else:
            popt = fit_constrained_function(dep_func, x, y, p0, method, bounds, constraints)
        
        # update the dependence function with fitted parameters
        dep_func.parameters = dict(zip(dep_func.parameters.keys(), popt))
        return dep_func
        

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
    def icdf(self, prob):
        """Inverse cumulative distribution function."""
        

    def fit(self, data, fixed=None, method="mle", weights=None):
        """Fit the distribution to the sampled data"""
        if method.lower() == "mle":
            self._fit_mle(data, fixed)
        elif method.lower() == "lsq":
            self._fit_lsq(data, fixed, weights)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Only Maximum-Likelihood-Estimation (mle) "
                             "and (weighted) least squares (lsq) are supported.")
        
    @abstractmethod
    def _fit_mle(self, data, fixed):
        """Fit the distribution using Maximum-Likelihood-Estimation."""
        
    @abstractmethod
    def _fit_lsq(self, data, fixed, weights):
        """Fit the distribution using (weighted) least squares."""



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
        return sts.weibull_min.cdf(x, c=self.k, loc=self.theta, scale=self.lambda_)

    def icdf(self, prob):
        return sts.weibull_min.ppf(prob, c=self.k, loc=self.theta, scale=self.lambda_)

    def pdf(self, x):
        return sts.weibull_min.pdf(x, c=self.k, loc=self.theta, scale=self.lambda_)
    
    def _fit_mle(self, samples, fixed):
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
        
    def _fit_lsq(self, data, fixed, weights):
        raise NotImplementedError()
        
class LogNormalDistribution(Distribution):
    
   
    def __init__(self, mu=0, sigma=1):
        
        self.mu = mu
        self.sigma = sigma  # shape
        #self.scale = math.exp(mu)
        
    @property
    def parameters(self):
        return {"mu" : self.mu,
                "sigma" : self.sigma}
    @property
    def _scale(self):
        return math.exp(self.mu)
    
    @_scale.setter
    def _scale(self, val):
        self.mu = math.log(val)
        
    def cdf(self, x):
        return sts.lognorm.cdf(x, s=self.sigma, scale=self._scale)

    def icdf(self, prob):
        return sts.lognorm.ppf(prob, s=self.sigma, scale=self._scale)

    def pdf(self, x):
        return sts.lognorm.pdf(x, s=self.sigma, scale=self._scale)
    
    def _fit_mle(self, samples, fixed):
        p0={"scale": self._scale, "sigma": self.sigma}
        
        fparams = {"floc" : 0}
        if fixed is not None:
            if "sigma" in fixed.keys():
                fparams["f0"] = fixed["sigma"]
            if "mu" in fixed.keys():
                fparams["fscale"] = math.exp(fixed["mu"])
        
        #scale0 = math.exp(p0["mu"])
        self.sigma, _, self._scale  = (
            sts.lognorm.fit(samples, p0["sigma"], scale=p0["scale"], **fparams)
             )
        #self.mu = math.log(self._scale)
        
        
    def _fit_lsq(self, data, fixed, weights):
        raise NotImplementedError()
        
class ExponentiatedWeibullDistribution(Distribution):
    """
    An exponentiated Weibull distribution.

    Note
    -----
    We use the parametrization that is also used in
    https://arxiv.org/pdf/1911.12835.pdf .
    """

    def __init__(self, alpha=1, beta=1, delta=1):
        self.alpha = alpha  # scale
        self.beta = beta  # shape
        self.delta = delta  # shape2
        # In scipy the order of the shape parameters is reversed:
        # a ^= delta
        # c ^= beta


    def cdf(self, x):
        return sts.exponweib.cdf(x, self.delta, self.beta, loc=0, scale=self.alpha)


    def icdf(self, prob):
        return sts.exponweib.ppf(prob, self.delta, self.beta, loc=0, scale=self.alpha)


    def pdf(self, x):
        #x = np.asarray(x, dtype=float)  # If x elements are int we cannot use np.nan .
        # This changes x which is unepexted behaviour!
        #x[x<=0] = np.nan  # To avoid warnings with negative and 0-values, use NaN.
        # TODO ensure for all pdf that no nan come up?
        x_greater_zero = np.where(x > 0, x, np.nan)
        _pdf = sts.exponweib.pdf(x_greater_zero, self.delta, self.beta, loc=0, scale=self.alpha)
        _pdf[np.isnan(_pdf)] = 0
        return _pdf

    
    def _fit_mle(self, samples, fixed):
        p0={"alpha": self.alpha, "beta": self.beta, "delta": self.delta}
    
        fparams = {}
        if fixed is not None:
            if "delta" in fixed.keys():
                fparams["f0"] = fixed["delta"]
            if "beta" in fixed.keys():
                fparams["f1"] = fixed["beta"]
            if "alpha" in fixed.keys():
                fparams["fscale"] = fixed["alpha"]
        
        self.delta, self.beta, _, self.alpha  = (
            sts.exponweib.fit(samples, p0["delta"], p0["beta"], loc=0, 
                                scale=p0["alpha"], **fparams)
             )
        
    def _fit_lsq(self, data, fixed, weights):
        # Based on Appendix A. in https://arxiv.org/pdf/1911.12835.pdf
        x = np.sort(np.asarray_chkfinite(data))
        if weights is None:
            weights = np.ones_like(x)
        elif isinstance(weights, str):
            if weights.lower() == "linear":
                weights = x ** 2 / np.sum(x ** 2)
            elif weights.lower() == "quadratic":
                weights = x ** 2 / np.sum(x ** 2)
            elif weights.lower() == "cubic":
                weights = x ** 2 / np.sum(x ** 2)
            else:
                raise ValueError(f"Unsupported value for weights={weights}.")
        else:
            try:
                _ = iter(weights)
                weights = np.asarray_chkfinite(weights)
            except TypeError:
                raise ValueError(f"Unsupported value for weights={weights}.")
                
        n = len(x)
        p = (np.arange(1, n + 1) - 0.5) / n
        
                
        if fixed is not None:
            raise NotImplementedError()
        else:
            delta0 = self.delta
            self.delta = fmin(self._wlsq_error, delta0, disp=False,
                              args=(x, p, weights))[0]
            
            self.alpha, self.beta = self._estimate_alpha_beta(self.delta, x, p, 
                                                              weights,
                                                              )
            
            
    @staticmethod   
    def _estimate_alpha_beta(delta, x, p, w, falpha=None, fbeta=None):
        
        # First, transform x and p to get a linear relationship.
        x_star = np.log10(x)
        p_star = np.log10(-np.log(1 - p ** (1 / delta)))
        
        
        # Estimate the parameters alpha_hat and beta_hat.
        p_star_bar = np.sum(w * p_star)
        x_star_bar = np.sum(w * x_star)
        b_hat_dividend = np.sum(w * p_star * x_star) - p_star_bar * x_star_bar
        b_hat_divisor = np.sum(w * p_star ** 2) - p_star_bar ** 2
        b_hat = b_hat_dividend / b_hat_divisor
        a_hat = x_star_bar - b_hat * p_star_bar
        alpha_hat = 10 ** a_hat
        beta_hat = b_hat_divisor / b_hat_dividend  # beta_hat = 1 / b_hat
        
        return alpha_hat, beta_hat
        
            
    @staticmethod
    def _wlsq_error(delta, x, p, w, return_alpha_beta=False, falpha=None, fbeta=None):

        # As x = 0 causes problems when x_star is calculated, zero-elements
        # are not considered in the parameter estimation.
        indices = np.nonzero(x)
        x = x[indices]
        p = p[indices]
        
        alpha_hat, beta_hat = ExponentiatedWeibullDistribution._estimate_alpha_beta(delta, x, p, w)
        
        # Compute the weighted least squares error.
        x_hat = alpha_hat * (-np.log(1 - p ** (1 / delta))) ** (1 / beta_hat)
        wlsq_error = np.sum(w * (x - x_hat) ** 2)

        return wlsq_error

            
        
        
                
        

                
        
        
        
        
        
        