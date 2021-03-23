import math
import copy

import numpy as np
import scipy.stats as sts

from abc import ABC, abstractmethod
from scipy.optimize import fmin

__all__ = ["WeibullDistribution", "LogNormalDistribution",
           "ExponentiatedWeibullDistribution"]

# The distributions parameters need to have an order, this order is defined by
# the parameters dict. As of Python 3.7 dicts officially keep their order of creation.
# So this version is a requirement.
# (Though the dict order might work as well in 3.6)



class ConditionalDistribution:

    
        """
        Conditional distributions for two or more (environmental) parameters 
        that are dependet on each other. 
        
        Parameters
        ----------
        distribution : Distribution
            Function, that calculates the probabilities of occurence for
            different possbile (environmental) events.
        parameters: 
           Probability distributions have parameters (1-3) that define its shape,
           location and scale. These parameters represent essential properties
           of the distribution.
        
        """


    
    def __init__(self, distribution, parameters):
        # allow setting fitting initials on class creation?
        self.distribution = distribution
        self.distribution_class = distribution.__class__
        self.param_names = distribution.parameters.keys()
        self.conditional_parameters = {}
        self.fixed_parameters = {}
        # TODO check that dependency functions are not duplicates

        # Check if the parameters dict contains keys/parameters that
        # are not known to the distribution.
        # (use set operations for that purpose)
        unknown_params = set(parameters).difference(self.param_names)
        if len(unknown_params) > 0:
            raise ValueError("Unknown param(s) in parameters."
                             f"Known params are {self.param_names}, "
                             f"but found {unknown_params}.")

        for par_name in self.param_names:
            # is the parameter defined as a dependence function?
            if par_name not in parameters:
                # if it is not a dependence function it must be fixed
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


    def _get_param_values(self, given):
        param_values = {}
        for par_name in self.param_names:
            if par_name in self.conditional_parameters.keys():
                param_values[par_name] = self.conditional_parameters[par_name](given)
            else:
                param_values[par_name] = self.fixed_parameters[par_name]

        return param_values

    def pdf(self, x, given):
        """Probability density function."""
        
        return self.distribution.pdf(x, **self._get_param_values(given))
    
    def cdf(self, x, given):
         """Cumulative distribution function."""
        
        return self.distribution.cdf(x, **self._get_param_values(given))
    
    def icdf(self, prob, given):
        """Inverse cumulative distribution function."""
        
        return self.distribution.icdf(prob, **self._get_param_values(given))
        
    def draw_sample(self, n, given):
         """Draw a random sample with length n."""
         
        return self.distribution.draw_sample(n, **self._get_param_values(given))
    
    def fit(self, data, conditioning_values, conditioning_interval_boundaries,
            fit_method="mle", weights=None):
        self.distributions_per_interval = []
        self.parameters_per_interval = []
        self.data_intervals = data
        self.conditioning_values = np.array(conditioning_values)
        self.conditioning_interval_boundaries = conditioning_interval_boundaries
        # Fit distribution to each interval.
        for interval_data in data:
            #dist = self.distribution_class()
            dist = copy.deepcopy(self.distribution)
            dist.fit(interval_data, fit_method, weights)
            self.distributions_per_interval.append(dist)
            self.parameters_per_interval.append(dist.parameters)
            
        # Fit dependence functions.
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
        
   Models the probabilities of occurence for different possbile 
   (environmental) events.
   """



    @property
    @abstractmethod
    def parameters(self):
        """
           Parameters of the probability distribution.
        
        """
        
        return {}

    @abstractmethod
    def cdf(self, x, *args, **kwargs):
        """Cumulative distribution function."""

    @abstractmethod
    def pdf(self, x, *args, **kwargs):
        """Probability density function."""

    @abstractmethod
    def icdf(self, prob, *args, **kwargs):
        """Inverse cumulative distribution function."""
        
    @abstractmethod
    def draw_sample(self, n,  *args, **kwargs):
        """Draw sample from distribution."""

    def fit(self, data, method="mle", weights=None):
        """Fit the distribution to the sampled data"""
            
        if method.lower() == "mle":
            self._fit_mle(data)
        elif method.lower() == "lsq" or method.lower() == "wlsq":
            self._fit_lsq(data, weights)
        else:
            raise ValueError(f"Unknown fit method '{method}'. "
                             "Only maximum likelihood estimation (keyword: mle) "
                             "and (weighted) least squares (keyword: (w)lsq) are supported.")
        
    @abstractmethod
    def _fit_mle(self, data):
        """Fit the distribution using maximum likelihood estimation."""
        
    @abstractmethod
    def _fit_lsq(self, data, weights):
        """Fit the distribution using (weighted) least squares."""


    @staticmethod
    def _get_rvs_size(n, pars):
        # Returns the size parameter for the scipy rvs method.
        # If there are any iterable pars it is a tuple,
        # otherwise n is returned.
        at_least_one_iterable = False
        par_length = 0
        for par in pars:
            try:
                _ = iter(par)
                at_least_one_iterable = True
                par_length = len(par)
            except TypeError:
                pass

        if at_least_one_iterable:
            return (n, par_length)
        else:
            return n



class WeibullDistribution(Distribution):
    
    
    def __init__(self, alpha=1, beta=1, gamma=0, f_alpha=None, f_beta=None,
                 f_gamma=None):
        
        # TODO set parameters to fixed values if provided
        self.alpha = alpha  # scale
        self.beta = beta  # shape
        self.gamma = gamma  # loc
        self.f_alpha = f_alpha
        self.f_beta = f_beta
        self.f_gamma = f_gamma
        
    @property
    def parameters(self):
        return {"alpha" : self.alpha,
                "beta" : self.beta,
                "gamma" : self.gamma}


    def _get_scipy_parameters(self, alpha, beta, gamma):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma
        return beta, gamma, alpha  # shape, loc, scale

    def cdf(self, x, alpha=None, beta=None, gamma=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.cdf(x, *scipy_par)


    def icdf(self, prob, alpha=None, beta=None, gamma=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.ppf(prob, *scipy_par)


    def pdf(self, x, alpha=None, beta=None, gamma=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.pdf(x, *scipy_par)


    def draw_sample(self, n, alpha=None, beta=None, gamma=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.weibull_min.rvs(*scipy_par, size=rvs_size)


    def _fit_mle(self, sample):
        p0={"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
        
        fparams = {}
        if self.f_beta is not None:
            fparams["f0"] = self.f_beta
        if self.f_gamma is not None:
            fparams["floc"] = self.f_gamma
        if self.f_alpha is not None:
            fparams["fscale"] = self.f_alpha

        self.beta, self.gamma, self.alpha  = (
            sts.weibull_min.fit(sample, p0["beta"], loc=p0["gamma"],
                                scale=p0["alpha"], **fparams)
             )
        
    def _fit_lsq(self, data, weights):
        raise NotImplementedError()
        

        
class LogNormalDistribution(Distribution):
    
   
    def __init__(self, mu=0, sigma=1, f_mu=None, f_sigma=None):
        
        self.mu = mu
        self.sigma = sigma  # shape
        self.f_mu = f_mu
        self.f_sigma = f_sigma
        #self.scale = math.exp(mu)
        
    @property
    def parameters(self):
        return {"mu" : self.mu,
                "sigma" : self.sigma}
    @property
    def _scale(self):
        return np.exp(self.mu)
    
    @_scale.setter
    def _scale(self, val):
        self.mu = np.log(val)

    def _get_scipy_parameters(self, mu, sigma):
        if mu is None:
            scale = self._scale
        else:
            scale = np.exp(mu)
        if sigma is None:
            sigma = self.sigma
        return sigma, 0, scale # shape, loc, scale
        
    def cdf(self, x, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.cdf(x, *scipy_par)

    def icdf(self, prob, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.ppf(prob, *scipy_par)

    def pdf(self, x, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.pdf(x, *scipy_par)

    def draw_sample(self, n, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.lognorm.rvs(*scipy_par, size=rvs_size)
    
    def _fit_mle(self, sample):
        p0={"scale": self._scale, "sigma": self.sigma}
        
        fparams = {"floc" : 0}
        
        if self.f_sigma is not None:
            fparams["f0"] = self.f_sigma
        if self.f_mu is not None:
            fparams["fscale"] = math.exp(self.f_mu)
        
        #scale0 = math.exp(p0["mu"])
        self.sigma, _, self._scale  = (
            sts.lognorm.fit(sample, p0["sigma"], scale=p0["scale"], **fparams)
             )
        #self.mu = math.log(self._scale)
        
        
    def _fit_lsq(self, data, weights):
        raise NotImplementedError()
        

        
        
class LogNormalNormFitDistribution(LogNormalDistribution):
    #https://en.wikipedia.org/wiki/Log-normal_distribution#Estimation_of_parameters
    
   
    def __init__(self, mu_norm=0, sigma_norm=1, f_mu_norm=None, f_sigma_norm=None):
        
        self.mu_norm = mu_norm
        self.sigma_norm = sigma_norm
        self.f_mu_norm = f_mu_norm
        self.f_sigma_norm = f_sigma_norm
        
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

    def _get_scipy_parameters(self, mu_norm, sigma_norm):
        if (mu_norm is None) != (sigma_norm is None):
            raise RuntimeError("mu_norm and sigma_norm have to be passed both or not at all")

        if mu_norm is None:
            scale = self._scale
            sigma = self.sigma
        else:
            sigma = self.calculate_sigma(mu_norm, sigma_norm)
            mu = self.calculate_mu(mu_norm, sigma_norm)
            scale = np.exp(mu)
        return sigma, 0, scale  # shape, loc, scale

    def cdf(self, x, mu_norm=None, sigma_norm=None):
        scipy_par = self._get_scipy_parameters(mu_norm, sigma_norm)
        return sts.lognorm.cdf(x, *scipy_par)

    def icdf(self, prob, mu_norm=None, sigma_norm=None):
        scipy_par = self._get_scipy_parameters(mu_norm, sigma_norm)
        return sts.lognorm.ppf(prob, *scipy_par)

    def pdf(self, x, mu_norm=None, sigma_norm=None):
        scipy_par = self._get_scipy_parameters(mu_norm, sigma_norm)
        return sts.lognorm.pdf(x, *scipy_par)

    def draw_sample(self, n,mu_norm=None, sigma_norm=None):
        scipy_par = self._get_scipy_parameters(mu_norm, sigma_norm)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.lognorm.rvs(*scipy_par, size=rvs_size)
    
     
    def _fit_mle(self, sample):
        

        if self.f_mu_norm is None:
            self.mu_norm = np.mean(sample)
        else:
            self.mu_norm = self.f_mu_norm
            
        if self.f_sigma_norm is None:
            self.sigma_norm = np.std(sample, ddof=1)
        else:
            self.sigma_norm = self.f_sigma_norm
        
        
    def _fit_lsq(self, data, weights):
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
                 f_delta=None):
        self.alpha = alpha  # scale
        self.beta = beta  # shape
        self.delta = delta  # shape2
        self.f_alpha = f_alpha
        self.f_beta = f_beta
        self.f_delta = f_delta
        # In scipy the order of the shape parameters is reversed:
        # a ^= delta
        # c ^= beta

    def _get_scipy_parameters(self, alpha, beta, delta):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if delta is None:
            delta = self.delta
        return delta, beta, 0, alpha # shape1, shape2, loc, scale

    def cdf(self, x, alpha=None, beta=None, delta=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        return sts.exponweib.cdf(x, *scipy_par)


    def icdf(self, prob, alpha=None, beta=None, delta=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        return sts.exponweib.ppf(prob, *scipy_par)

    def pdf(self, x, alpha=None, beta=None, delta=None):
        #x = np.asarray(x, dtype=float)  # If x elements are int we cannot use np.nan .
        # This changes x which is unepexted behaviour!
        #x[x<=0] = np.nan  # To avoid warnings with negative and 0-values, use NaN.
        # TODO ensure for all pdf that no nan come up?
        x_greater_zero = np.where(x > 0, x, np.nan)
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        _pdf = sts.exponweib.pdf(x_greater_zero, *scipy_par)
        if _pdf.shape == (): # x was scalar
            if np.isnan(_pdf):
                _pdf = 0
        else:          
            _pdf[np.isnan(_pdf)] = 0
        return _pdf

    def draw_sample(self, n, alpha=None, beta=None, delta=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.exponweib.rvs(*scipy_par, size=rvs_size)

    
    def _fit_mle(self, sample):
        p0={"alpha": self.alpha, "beta": self.beta, "delta": self.delta}
    
        fparams = {"floc" : 0}
        
        if self.f_delta is not None:
            fparams["f0"] = self.f_delta 
        if self.f_beta is not None:
            fparams["f1"] = self.f_beta 
        if self.f_alpha is not None:
             fparams["fscale"] = self.f_alpha
                
        self.delta, self.beta, _, self.alpha  = (
            sts.exponweib.fit(sample, p0["delta"], p0["beta"],
                                scale=p0["alpha"], **fparams)
             )
        
    def _fit_lsq(self, data, weights):
        # Based on Appendix A. in https://arxiv.org/pdf/1911.12835.pdf
        x = np.sort(np.asarray_chkfinite(data))
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
    

            
        
        
                
        

                
        
        
        
        
        
        