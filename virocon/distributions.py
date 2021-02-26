import math
import copy

import numpy as np
import scipy.stats as sts

from abc import ABC, abstractmethod
from scipy.optimize import fmin


# The distributions parameters need to have an order, this order is defined by
# the parameters dict. As of Python 3.7 dicts officially keep their order of creation.
# So this version is a requirement.
# (Though the dict order might work as well in 3.6)

# TODO possibly use something like
# https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
class ConditionalDistribution():
    
    def __init__(self, distribution, parameters):
        # allow setting fitting initials on class creation?
        self.distribution = distribution
        self.distribution_class = distribution.__class__
        self.param_names = distribution.parameters.keys()
        self.conditional_parameters = {}
        self.fixed_parameters = {}
        # TODO check that dependency functions are not duplicates
        
        unknown_params = set(parameters).difference(self.param_names)
        if len(unknown_params) > 0:
            raise ValueError("Unknown param(s) in parameters."
                             f"Known params are {self.param_names}, "
                             f"but found {unknown_params}.")

        for par_name in self.param_names:
            # is the parameter defined as a dependence function?
            if par_name not in parameters:
                # if it is not dependence function it must be fixed
                if getattr(distribution, f"f_{par_name}") is None:
                    raise ValueError("Parameters of the distribution must be "
                                     "either defined by a dependence function "                                     
                                     f"or fixed, but {par_name} was not defined.")
                else:
                     self.fixed_parameters[par_name] = getattr(distribution, f"f_{par_name}")
            else:
                # if it is a dependence function it must not be fixed
                if getattr(distribution, f"f_{par_name}") is not None:
                    raise ValueError("Parameters can be defined by a "
                                     "dependence function or by being fixed. "
                                     f"But for parameter {par_name} both where given."
                                     )
                else:
                    self.conditional_parameters[par_name] = parameters[par_name]

                
        


        
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
            #dist = self.distribution_class()
            dist = copy.deepcopy(self.distribution)
            dist.fit(interval_data)
            self.distributions_per_interval.append(dist)
            self.parameters_per_interval.append(dist.parameters)
            
        # fit dependence functions
        fitted_dependence_functions = {}
        for par_name, dep_func in self.conditional_parameters.items():
            x = self.conditioning_values
            y = [params[par_name] for params in self.parameters_per_interval]
            dep_func.fit(x, y)
            fitted_dependence_functions[par_name] = dep_func
            
        self.conditional_parameters = fitted_dependence_functions


class Distribution(ABC):
    """
    Abstract base class for distributions.
    """

    @property
    @abstractmethod
    def parameters(self):
        return {}

    @abstractmethod
    def cdf(self, x,):
        """Cumulative distribution function."""

    @abstractmethod
    def pdf(self, x):
        """Probability density function."""

    @abstractmethod
    def icdf(self, prob):
        """Inverse cumulative distribution function."""
        

    def fit(self, data):
        """Fit the distribution to the sampled data"""
        method = self.fit_method
            
        if method.lower() == "mle":
            self._fit_mle(data)
        elif method.lower() == "lsq" or method.lower() == "wlsq":
            self._fit_lsq(data)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Only Maximum-Likelihood-Estimation (mle) "
                             "and (weighted) least squares (lsq) are supported.")
        
    @abstractmethod
    def _fit_mle(self, data):
        """Fit the distribution using Maximum-Likelihood-Estimation."""
        
    @abstractmethod
    def _fit_lsq(self, data):
        """Fit the distribution using (weighted) least squares."""
        
    @abstractmethod
    def draw_sample(self, n):
        """Draw samples from distribution."""



class WeibullDistribution(Distribution):
    
    
    def __init__(self, lambda_=1, k=1, theta=0, f_lambda_=None, f_k=None, 
                 f_theta=None, fit_method="mle", weights=None):
        
        # TODO set parameters to fixed values if provided
        self.lambda_ = lambda_  # scale
        self.k = k  # shape
        self.theta = theta  # loc
        self.f_lambda_ = f_lambda_ 
        self.f_k = f_k
        self.f_theta = f_theta
        self.fit_method = fit_method
        self.weights = weights
        
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
    
    def _fit_mle(self, samples):
        p0={"lambda_": self.lambda_, "k": self.k, "theta": self.theta}
        
        fparams = {}
        if self.f_k is not None:
            fparams["f0"] = self.f_k
        if self.f_theta is not None:
            fparams["floc"] = self.f_theta
        if self.f_lambda_ is not None:
            fparams["fscale"] = self.f_lambda_
        
        self.k, self.theta, self.lambda_  = (
            sts.weibull_min.fit(samples, p0["k"], loc=p0["theta"], 
                                scale=p0["lambda_"], **fparams)
             )
        
    def _fit_lsq(self, data):
        raise NotImplementedError()
        
    def draw_sample(self, n):
        return sts.weibull_min.rvs(size=n, c=self.k, loc=self.theta, scale=self.lambda_)
        
class LogNormalDistribution(Distribution):
    
   
    def __init__(self, mu=0, sigma=1, f_mu=None, f_sigma=None, fit_method="mle",
                 weights=None):
        
        self.mu = mu
        self.sigma = sigma  # shape
        self.f_mu = f_mu
        self.f_sigma = f_sigma
        self.fit_method = fit_method
        self.weights = weights
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
    
    def _fit_mle(self, samples):
        p0={"scale": self._scale, "sigma": self.sigma}
        
        fparams = {"floc" : 0}
        
        if self.f_sigma is not None:
            fparams["f0"] = self.f_sigma
        if self.f_mu is not None:
            fparams["fscale"] = math.exp(self.f_mu)
        
        #scale0 = math.exp(p0["mu"])
        self.sigma, _, self._scale  = (
            sts.lognorm.fit(samples, p0["sigma"], scale=p0["scale"], **fparams)
             )
        #self.mu = math.log(self._scale)
        
        
    def _fit_lsq(self, data):
        raise NotImplementedError()
        
    def draw_sample(self, n):
        return sts.lognorm.rvs(size=n, s=self.sigma, scale=self._scale)
        
        
class LogNormalNormFitDistribution(LogNormalDistribution):
    #https://en.wikipedia.org/wiki/Log-normal_distribution#Estimation_of_parameters
    
   
    def __init__(self, mu_norm=0, sigma_norm=1, f_mu_norm=None, f_sigma_norm=None, 
                 fit_method="mle", weights=None):
        
        self.mu_norm = mu_norm
        self.sigma_norm = sigma_norm
        self.f_mu_norm = f_mu_norm
        self.f_sigma_norm = f_sigma_norm
        self.fit_method = fit_method
        self.weights = weights
        
    @property
    def parameters(self):
        return {"mu_norm" : self.mu_norm,
                "sigma_norm" : self.sigma_norm}

    @property
    def mu(self):
        return self.calculate_mu(self.mu_norm, self.sigma_norm)
    
    @staticmethod
    def calculate_mu(mu_norm, sigma_norm):
        return np.log(mu_norm / np.sqrt(1 + sigma_norm**2 / mu_norm**2))
        # return np.log(mu_norm**2 * np.sqrt(1 / (sigma_norm**2 + mu_norm**2)))
    
    @property
    def sigma(self):
        return self.calculate_sigma(self.mu_norm, self.sigma_norm)
    
    @staticmethod
    def calculate_sigma(mu_norm, sigma_norm):
        # return np.sqrt(np.log(1 + sigma_norm**2 / mu_norm**2))
        return np.sqrt(np.log(1 + (sigma_norm**2 / mu_norm**2)))
    
     
    def _fit_mle(self, samples):
        

        if self.f_mu_norm is None:
            self.mu_norm = np.mean(samples)
        else:
            self.mu_norm = self.f_mu_norm
            
        if self.f_sigma_norm is None:
            self.sigma_norm = np.std(samples, ddof=1)
        else:
            self.sigma_norm = self.f_sigma_norm
        
        # self.mu_norm, self.sigma_norm = sts.norm.fit(samples)
        
        
    def _fit_lsq(self, data):
        raise NotImplementedError()
        
        
class ExponentiatedWeibullDistribution(Distribution):
    """
    An exponentiated Weibull distribution.

    Note
    -----
    We use the parametrization that is also used in
    https://arxiv.org/pdf/1911.12835.pdf .
    """
    
    @property
    def parameters(self):
        return {"alpha" : self.alpha,
                "beta" : self.beta,
                "delta" : self.delta}

    def __init__(self, alpha=1, beta=1, delta=1, f_alpha=None, f_beta=None, 
                 f_delta=None, fit_method="mle", weights=None):
        self.alpha = alpha  # scale
        self.beta = beta  # shape
        self.delta = delta  # shape2
        self.f_alpha = f_alpha
        self.f_beta = f_beta
        self.f_delta = f_delta
        self.fit_method = fit_method
        self.weights = weights
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
        if _pdf.shape == (): # x was scalar
            if np.isnan(_pdf):
                _pdf = 0
        else:          
            _pdf[np.isnan(_pdf)] = 0
        return _pdf

    
    def _fit_mle(self, samples):
        p0={"alpha": self.alpha, "beta": self.beta, "delta": self.delta}
    
        fparams = {"floc" : 0}
        
        if self.f_delta is not None:
            fparams["f0"] = self.f_delta 
        if self.f_beta is not None:
            fparams["f1"] = self.f_beta 
        if self.f_alpha is not None:
             fparams["fscale"] = self.f_alpha
                
        self.delta, self.beta, _, self.alpha  = (
            sts.exponweib.fit(samples, p0["delta"], p0["beta"], 
                                scale=p0["alpha"], **fparams)
             )
        
    def _fit_lsq(self, data):
        # Based on Appendix A. in https://arxiv.org/pdf/1911.12835.pdf
        x = np.sort(np.asarray_chkfinite(data))
        weights = self.weights
        if weights is None:
            weights = np.ones_like(x)
        elif isinstance(weights, str):
            if weights.lower() == "linear":
                weights = x / np.sum(x)
            elif weights.lower() == "quadratic":
                weights = x ** 2 / np.sum(x ** 2)
            elif weights.lower() == "cubic":
                weights = x ** 3 / np.sum(x ** 3)
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
        
        fixed = {}
        if self.f_alpha is not None:
            fixed["alpha"] = self.f_alpha
        if self.f_beta is not None:
            fixed["beta"] = self.f_beta
        if self.f_delta is not None:
            fixed["delta"] = self.f_delta
        if len(fixed) == 0:
            fixed = None
                
        if fixed is not None:
            if "delta" in fixed and not ("alpha" in fixed or "beta" in fixed):
                self.delta = fixed["delta"]
                self.alpha, self.beta = self._estimate_alpha_beta(self.delta, x, p, 
                                                                  weights,
                                                                  )
            else:
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
    
    def draw_sample(self, n):
        return sts.exponweib.rvs(self.delta, self.beta, loc=0, scale=self.alpha, size=n)
            
        
        
                
        

                
        
        
        
        
        
        