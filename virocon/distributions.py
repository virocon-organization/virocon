"""
Distributions for single variables.
"""

import math
import copy

import numpy as np
import scipy.stats as sts

from abc import ABC, abstractmethod
from scipy.optimize import fmin

__all__ = [
    "WeibullDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "ExponentiatedWeibullDistribution",
    "GeneralizedGammaDistribution",
    "VonMisesDistribution"
]

# The distributions parameters need to have an order, this order is defined by
# the parameters dict. As of Python 3.7 dicts officially keep their order of creation.
# So this version is a requirement.
# (Though the dict order might work as well in 3.6)


class ConditionalDistribution:
    """
    A conditional probability distribution.
    
    The conditional distribution uses a Distribution as template and 
    dynamically alters its parameters to model the dependence. The 
    ConditionalDistribution wraps another distribution. When a method of 
    the ConditionalDistribution is called it first computes the distributions 
    parameters at given and then calls the corresponding method of the 
    distribution with these parameters. Usually the parameters are defined by 
    dependence functions of the form dep_func(given) -> param_val.
    
    Parameters
    ----------
    distribution : Distribution
        The distribution used as template. Its parameters can be replaced with 
        dependence functions to model the dependency.
    parameters: float
       A dictionary describing the parameters of distribution. The keys are 
       the parameter names, the values are the dependence functions. Every 
       parameter that is not fixed in distribution has to be set here.

    Attributes
    ----------
    distribution_class : type
        The class of the distribution used.
    param_names : list-like
        Names of the parameters of the distribution.
    conditional_parameters : dict
        Dictionary of dependence functions for conditional parameters. Parameter names as keys.
    fixed_parameters : dict
        Values of the fixed parameters. The fixed parameters do not change, 
        even when fitting them. Parameters as keys.
    distributions_per_interval : list
        Instances of distribution fitted to intervals
    parameters_per_interval : list of dict
        Values of the parameters of the distribution function. Parameter names as keys.
    data_intervals : list of array
        The data that was used to fit the distribution. Split into intervals.
    conditioning_values : array_like
        Realizations of the conditioning variable that where used for fitting.
    conditioning_interval_boundaries : list of tuple
        Boundaries of the intervals the data of the conditioning variable
        was split into.
    """

    def __init__(self, distribution, parameters):
        # allow setting fitting initials on class creation?
        self.distribution = distribution
        self.distribution_class = distribution.__class__
        self.param_names = distribution.parameters.keys()
        self.conditional_parameters = {}
        self.fixed_parameters = {}
        self.conditioning_values = None
        # TODO check that dependency functions are not duplicates

        # Check if the parameters dict contains keys/parameters that
        # are not known to the distribution.
        # (use set operations for that purpose)
        unknown_params = set(parameters).difference(self.param_names)
        if len(unknown_params) > 0:
            raise ValueError(
                "Unknown param(s) in parameters."
                f"Known params are {self.param_names}, "
                f"but found {unknown_params}."
            )

        for par_name in self.param_names:
            # is the parameter defined as a dependence function?
            if par_name not in parameters:
                # if it is not a dependence function it must be fixed
                if getattr(distribution, f"f_{par_name}") is None:
                    raise ValueError(
                        "Parameters of the distribution must be "
                        "either defined by a dependence function "
                        f"or fixed, but {par_name} was not defined."
                    )
                else:
                    self.fixed_parameters[par_name] = getattr(
                        distribution, f"f_{par_name}"
                    )
            else:
                # if it is a dependence function it must not be fixed
                if getattr(distribution, f"f_{par_name}") is not None:
                    raise ValueError(
                        "Parameters can be defined by a "
                        "dependence function or by being fixed. "
                        f"But for parameter {par_name} both where given."
                    )
                else:
                    self.conditional_parameters[par_name] = parameters[par_name]

    def __repr__(self):
        dist = "Conditional" + self.distribution_class.__name__
        fixed_params = ", ".join(
            [
                f"f_{par_name}={par_value}"
                for par_name, par_value in self.fixed_parameters.items()
            ]
        )
        cond_params = ", ".join(
            [
                f"{par_name}={repr(par_value)}"
                for par_name, par_value in self.conditional_parameters.items()
            ]
        )
        combined_params = fixed_params + ", " + cond_params
        combined_params = combined_params.strip(", ")
        return f"{dist}({combined_params})"

    def _get_param_values(self, given):
        param_values = {}
        for par_name in self.param_names:
            if par_name in self.conditional_parameters.keys():
                param_values[par_name] = self.conditional_parameters[par_name](given)
            else:
                param_values[par_name] = self.fixed_parameters[par_name]

        return param_values

    def pdf(self, x, given):
        """
        Probability density function for the described random variable.

        With x, value(s) from the sample space of this random variable and 
        given value(s) from the sample space of the conditioning random 
        variable, pdf(x, given) returns the probability density function at x 
        conditioned on given.
           
        Parameters
        ----------
        x : array_like
            Points at which the pdf is evaluated.
            Shape: 1- dimensional.
        
        given : float or array_like
           The conditioning value of the conditioning variable i.e. the
           y in x|y.  
           Shape: 1-dimensional. Same size as x.
            
        Returns
        -------
        ndarray
            Probability densities at x conditioned on given.
            Shape: 1- dimensional. Same size as x.
        
        
        """

        return self.distribution.pdf(x, **self._get_param_values(given))

    def cdf(self, x, given):
        """
        Cumulative distribution function for the described random variable.

        With x, a realization of this random variable and given a realisation 
        of the conditioning random variable, cdf(x, given) returns the 
        cumulative distribution function at x conditioned on given. 
     
        Parameters
        ----------
        x : array_like
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        
        given : float or array_like
           The conditioning value of the conditioning variable i.e. the
           y in x|y.  
           Shape: 1-dimensional. Same size as x.
   
        Returns
        -------
        ndarray
            Cumulative distribution function evaluated at x.
            Shape: 1-dimensional. Same size as x.
        
        """

        return self.distribution.cdf(x, **self._get_param_values(given))

    def icdf(self, prob, given):
        """
        Inverse cumulative distribution function.
        
        Calculate the inverse cumulative distribution function. Also known as quantile or 
        percent-point function. With x, a realization of this random variable 
        and given a realisation of the conditioning random variable, 
        icdf(x, given) returns the inverse cumulative distribution function at 
        x conditioned on given.
        
        
        Parameters
        ----------
        prob : 
            Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        
        given : float or array_like
           The conditioning value of the conditioning variable i.e. the
           y in x|y.  
           Shape: 1-dimensional. Same size as prob.
            
        Returns
        -------
        ndarray or float
            Inverse cumulative distribution function evaluated at given 
            probabilities conditioned on given.
            Shape: 1-dimensional. Same size as prob.
        
        """

        return self.distribution.icdf(prob, **self._get_param_values(given))

    def draw_sample(self, n, given):
        """
        Draw a random sample of size n, conditioned on given.
        
        
        Parameters
        ----------
        n : float
            Number of observations that shall be drawn.
        
        given : float or array_like
           The conditioning value of the conditioning variable i.e. the
           y in x|y.  
           Shape: TODO
            
        Returns
        -------
        ndarray or float
            Sample of the requested size conditioned on given. 
        
        """

        return self.distribution.draw_sample(n, **self._get_param_values(given))

    def fit(
        self,
        data,
        conditioning_values,
        conditioning_interval_boundaries,
        method=None,
        weights=None,
    ):
        """
        Fit statistical distribution to data.
        
        Method of estimating the parameters of a probability distribution to
        given data.
        
        Parameters
        ----------
        data : list of array
            The data that should be used to fit the distribution.
            Realizations of the distributions variable split into intervals. 
            One array for each interval containing the data in that interval.
        conditioning_values : array_like
            Realizations of the conditioning variable i.e. the y in x|y.  
            One value for each interval in data.
        conditioning_interval_boundaries : list of tuple
            Boundaries of the intervals the data of the conditioning variable
            was split into.
            One 2-element tuple for each interval in data.
        method : str, optional
            The method used to fit the distributions (self.distribution) for each interval.
            Defaults to the distributions default.
        weights :
            The weights used to fit the distributions (self.distribution) for each interval,
            when method is 'wlsq' = weighted least squares.

        """

        self.distributions_per_interval = []
        self.parameters_per_interval = []
        self.data_intervals = data
        self.conditioning_values = np.array(conditioning_values)
        self.conditioning_interval_boundaries = conditioning_interval_boundaries
        # Fit distribution to each interval.
        for interval_data in data:
            # dist = self.distribution_class()
            dist = copy.deepcopy(self.distribution)
            dist.fit(interval_data, method, weights)
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
         
    Models the probabilities of occurrence for different possible
    (environmental) events.
    
    """

    def __repr__(self):
        dist_name = self.__class__.__name__
        param_names = self.parameters.keys()
        set_params = {}
        for par_name in param_names:
            f_attr = getattr(self, f"f_{par_name}")
            if f_attr is not None:
                set_params[f"f_{par_name}"] = f_attr
            else:
                set_params[par_name] = getattr(self, par_name)

        params = ", ".join(
            [f"{par_name}={par_value}" for par_name, par_value in set_params.items()]
        )

        return f"{dist_name}({params})"

    @property
    @abstractmethod
    def parameters(self):

        """
        Parameters of the probability distribution.
        
        Dict of the form: {"<parameter_name>" : <parameter_value>, ...}
        
        """

        return {}

    @abstractmethod
    def cdf(self, x, *args, **kwargs):
        """
        Cumulative distribution function.
        
        """

    @abstractmethod
    def pdf(self, x, *args, **kwargs):
        """
        Probability density function.
        
        """

    @abstractmethod
    def icdf(self, prob, *args, **kwargs):
        """
        Inverse cumulative distribution function.
        
        """

    @abstractmethod
    def draw_sample(self, n, *args, **kwargs):
        """
        Draw a random sample of length n.
       
        """

    def fit(self, data, method="mle", weights=None):
        """
        Fit the distribution to the sampled data.

        data : array_like
            The observed data to fit the distribution.
        method : str, optional
            The method used for fitting. Defaults to 'mle' = maximum-likelihood estimation.
            Other options are 'lsq' / 'wlsq' for (weighted) least squares.
        weights : None, str, array_like,
            The weights to use for weighted least squares fitting. Ignored otherwise.
            Defaults to None = equal weights.
            Can be either an array_like with one weight for each point in data or a str.
            Valid options for str are: 'linear', 'quadratic', 'cubic'.
        """

        if method.lower() == "mle":
            self._fit_mle(data)
        elif method.lower() == "lsq" or method.lower() == "wlsq":
            self._fit_lsq(data, weights)
        else:
            raise ValueError(
                f"Unknown fit method '{method}'. "
                "Only maximum likelihood estimation (keyword: mle) "
                "and (weighted) least squares (keyword: (w)lsq) are supported."
            )

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
    """
    A weibull distribution. 
   
    The distributions probability density function is given by [1]_ :
    
    :math:`f(x) = \\frac{\\beta}{\\alpha} \\left (\\frac{x-\\gamma}{\\alpha} \\right)^{\\beta -1} \\exp \\left[-\\left( \\frac{x-\\gamma}{\\alpha} \\right)^{\\beta} \\right]`
    
    Parameters
    ----------
    alpha : float
        Scale parameter of the weibull distribution. Defaults to 1.
    beta : float
        Shape parameter of the weibull distribution. Defaults to 1.
    gamma : float
        Location parameter of the weibull distribution (3-parameter weibull
        distribution). Defaults to 0.
    f_alpha : float
        Fixed scale parameter of the weibull distribution (e.g. given physical
        parameter). If this parameter is set, alpha is ignored. The fixed
        parameter does not change, even when fitting the distribution. Defaults to None.
    f_beta : float
       Fixed shape parameter of the weibull distribution (e.g. given physical
       parameter). If this parameter is set, beta is ignored. The fixed parameter
       does not change, even when fitting the distribution. Defaults to None.
    f_gamma : float
        Fixed location parameter of the weibull distribution (e.g. given physical
        parameter). If this parameter is set, gamma is ignored. The fixed
        parameter does not change, even when fitting the distribution. Defaults to None.

    References
    ----------
    .. [1] Haselsteiner, A.F.; Ohlendorf, J.H.; Wosniok, W.; Thoben, K.D.(2017)
        Deriving environmental contours from highest density regions.  
        Coastal Engineering 123 (2017) 42–51.
        
    """

    def __init__(
        self, alpha=1, beta=1, gamma=0, f_alpha=None, f_beta=None, f_gamma=None
    ):
        self.alpha = alpha if f_alpha is None else f_alpha  # scale
        self.beta = beta if f_beta is None else f_beta  # shape
        self.gamma = gamma if f_gamma is None else f_gamma  # loc
        self.f_alpha = f_alpha
        self.f_beta = f_beta
        self.f_gamma = f_gamma

    @property
    def parameters(self):
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

    def _get_scipy_parameters(self, alpha, beta, gamma):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma
        return beta, gamma, alpha  # shape, loc, scale

    def cdf(self, x, alpha=None, beta=None, gamma=None):
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        alpha : float, optional
            The scale parameter. Defaults to self.alpha.
        beta : float, optional
            The shape parameter. Defaults to self.beta.
        gamma: float, optional
            The location parameter . Defaults to self.gamma.
        
        """

        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.cdf(x, *scipy_par)

    def icdf(self, prob, alpha=None, beta=None, gamma=None):
        """
        Inverse cumulative distribution function.
        
        Parameters
        ----------
        prob : array_like
            Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        alpha : float, optional
            The scale parameter. Defaults to self.aplha .
        beta : float, optional
            The shape parameter. Defaults to self.beta.
        gamma: float, optional
            The location parameter . Defaults to self.gamma.
        
        """

        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.ppf(prob, *scipy_par)

    def pdf(self, x, alpha=None, beta=None, gamma=None):
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the pdf is evaluated.
            Shape: 1-dimensional.
        alpha_ : float, optional
            The scale parameter. Defaults to self.alpha.
        beta : float, optional
            The shape parameter. Defaults to self.beta.
        gamma: float, optional
            The location parameter . Defaults to self.gamma.
        
        """

        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        return sts.weibull_min.pdf(x, *scipy_par)

    def draw_sample(self, n, alpha=None, beta=None, gamma=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, gamma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.weibull_min.rvs(*scipy_par, size=rvs_size)

    def _fit_mle(self, sample):
        p0 = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

        fparams = {}
        if self.f_beta is not None:
            fparams["f0"] = self.f_beta
        if self.f_gamma is not None:
            fparams["floc"] = self.f_gamma
        if self.f_alpha is not None:
            fparams["fscale"] = self.f_alpha

        self.beta, self.gamma, self.alpha = sts.weibull_min.fit(
            sample, p0["beta"], loc=p0["gamma"], scale=p0["alpha"], **fparams
        )

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    """
    A Lognormal Distribution. 
    
    The distributions probability density function is given by [2]_: 
    
    :math:`f(x) = \\frac{1}{x\\widetilde{\\sigma} \\sqrt{2\\pi}}\\exp \\left[ \\frac{-(\\ln x - \\widetilde{\\mu})^2}{2\\widetilde{\\sigma}^2}\\right]`
     
    
    Parameters
    ----------
    mu : float
        Mean parameter of the corresponding normal distribution. 
        Defaults to 0.
    sigma : float
        Standard deviation of the corresponding normal distribution. 
        Defaults to 1.
    f_mu : float
        Fixed parameter mu of the lognormal distribution (e.g. given physical
        parameter). If this parameter is set, mu is ignored. The fixed 
        parameter does not change, even when fitting the distribution. Defaults to None.
    f_sigma : float
       Fixed parameter sigma of the lognormal distribution (e.g. given 
       physical parameter). If this parameter is set, sigma is ignored. The
       fixed parameter does not change, even when fitting the distribution. Defaults to None.
    
    References
    ----------
    .. [2] Forbes, C.; Evans, M.; Hastings, N; Peacock, B. (2011)
        Statistical Distributions, 4th Edition, Published by 
        John Wiley & Sons, Inc., Hoboken, New Jersey., 
        Pages 131-132
    """

    def __init__(self, mu=0, sigma=1, f_mu=None, f_sigma=None):

        self.mu = mu if f_mu is None else f_mu
        self.sigma = sigma if f_sigma is None else f_sigma  # shape
        self.f_mu = f_mu
        self.f_sigma = f_sigma
        # self.scale = math.exp(mu)

    @property
    def parameters(self):
        return {"mu": self.mu, "sigma": self.sigma}

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
        return sigma, 0, scale  # shape, loc, scale

    def cdf(self, x, mu=None, sigma=None):
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        mu : float, optional
            The variance parameter. Defaults to self.mu .
        sigma : float, optional
            The shape parameter. Defaults to self.sigma .
        
        """

        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.cdf(x, *scipy_par)

    def icdf(self, prob, mu=None, sigma=None):
        """
        Inverse cumulative distribution function.
        
        Parameters
        ----------
        prob : Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        mu : float, optional
            The variance parameter. Defaults to self.mu .
        sigma : float, optional
            The shape parameter. Defaults to self.sigma .
        
        """

        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.ppf(prob, *scipy_par)

    def pdf(self, x, mu=None, sigma=None):
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the pdf is evaluated.
            Shape: 1-dimensional.
        mu : float, optional
            The variance parameter. Defaults to self.mu .
        sigma : float, optional
            The shape parameter. Defaults to self.sigma .
        
        """

        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.lognorm.pdf(x, *scipy_par)

    def draw_sample(self, n, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.lognorm.rvs(*scipy_par, size=rvs_size)

    def _fit_mle(self, sample):
        p0 = {"scale": self._scale, "sigma": self.sigma}

        fparams = {"floc": 0}

        if self.f_sigma is not None:
            fparams["f0"] = self.f_sigma
        if self.f_mu is not None:
            fparams["fscale"] = math.exp(self.f_mu)

        # scale0 = math.exp(p0["mu"])
        self.sigma, _, self._scale = sts.lognorm.fit(
            sample, p0["sigma"], scale=p0["scale"], **fparams
        )
        # self.mu = math.log(self._scale)

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()


class NormalDistribution(Distribution):
    """
    A Normal (Gaussian) Distribution. 
    
    The distributions probability density function is given by [3]_: 
    
    :math:`f(x) = \\frac{1}{{\\sigma} \\sqrt{2\\pi}} \\exp \\left( - \\frac{( x - \\mu)^2}{2\\sigma^2}\\right)`
     
    
    Parameters
    ----------
    mu : float
        Location parameter, the mean.
        Defaults to 0.
    sigma : float
        Scale parameter, the standard deviation.
        Defaults to 1.
    f_mu : float
        Fixed parameter mu of the normal distribution (e.g. given physical
        parameter). If this parameter is set, mu is ignored. The fixed 
        parameter does not change, even when fitting the distribution. Defaults to None.
    f_sigma : float
       Fixed parameter sigma of the normal distribution (e.g. given 
       physical parameter). If this parameter is set, sigma is ignored. The fixed
       parameter does not change, even when fitting the distribution. Defaults to None.
    
    References
    ----------
    .. [3] Forbes, C.; Evans, M.; Hastings, N; Peacock, B. (2011)
        Statistical Distributions, 4th Edition, Published by 
        John Wiley & Sons, Inc., Hoboken, New Jersey., 
        Page 143
    """

    def __init__(self, mu=0, sigma=1, f_mu=None, f_sigma=None):

        self.mu = mu if f_mu is None else f_mu  # location
        self.sigma = sigma if f_sigma is None else f_sigma  # scale
        self.f_mu = f_mu
        self.f_sigma = f_sigma

    @property
    def parameters(self):
        return {"mu": self.mu, "sigma": self.sigma}

    def _get_scipy_parameters(self, mu, sigma):
        if mu is None:
            loc = self.mu
        else:
            loc = mu
        if sigma is None:
            scale = self.sigma
        else:
            scale = self.sigma
        return loc, scale  # loc, scale

    def cdf(self, x, mu=None, sigma=None):
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.norm.cdf(x, *scipy_par)

    def icdf(self, prob, mu=None, sigma=None):
        """
        Inverse cumulative distribution function.
        
        Parameters
        ----------
        prob : Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.norm.ppf(prob, *scipy_par)

    def pdf(self, x, mu=None, sigma=None):
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the pdf is evaluated.
            Shape: 1-dimensional.
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(mu, sigma)
        return sts.norm.pdf(x, *scipy_par)

    def draw_sample(self, n, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(mu, sigma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.norm.rvs(*scipy_par, size=rvs_size)

    def _fit_mle(self, sample):
        p0 = {"loc": self.mu, "scale": self.sigma}

        fparams = {}

        if self.f_mu is not None:
            fparams["floc"] = self.f_mu
        if self.f_sigma is not None:
            fparams["fscale"] = self.f_sigma

        self.mu, self.sigma = sts.norm.fit(
            sample, loc=p0["loc"], scale=p0["scale"], **fparams
        )

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()


class LogNormalNormFitDistribution(LogNormalDistribution):
    # https://en.wikipedia.org/wiki/Log-normal_distribution#Estimation_of_parameters
    """
    A Lognormal Distribution. 
    The distributions probability density function is given by: 
    
    :math:`f(x) = \\frac{1}{x\\widetilde{\\sigma} \\sqrt{2\\pi}}\\exp \\left[ \\frac{-(\\ln x - \\widetilde{\\mu})^2}{2\\widetilde{\\sigma}^2}\\right]`
     
    
    Parameters
    ----------
    mu : float
        Mean parameter of the corresponding normal distribution. 
        Defaults to 0.
    sigma : float
        Variance parameter of the corresponding normal distribution. 
        Defaults to 1.
    f_mu : float
        Fixed parameter mu of the lognormal distribution (e.g. given physical
        parameter). If this parameter is set, mu is ignored. The fixed 
        parameter does not change, even when fitting the distribution. Defaults to None.
    f_sigma : float
       Fixed parameter sigma of the lognormal distribution (e.g. given 
       physical parameter). If this parameter is set, sigma is ignored. The
       fixed parameter does not change, even when fitting the distribution.
       Defaults to None. 
    
    """

    def __init__(self, mu_norm=0, sigma_norm=1, f_mu_norm=None, f_sigma_norm=None):

        self.mu_norm = mu_norm if f_mu_norm is None else f_mu_norm
        self.sigma_norm = sigma_norm if f_sigma_norm is None else f_sigma_norm
        self.f_mu_norm = f_mu_norm
        self.f_sigma_norm = f_sigma_norm

    @property
    def parameters(self):
        return {"mu_norm": self.mu_norm, "sigma_norm": self.sigma_norm}

    @property
    def mu(self):
        return self.calculate_mu(self.mu_norm, self.sigma_norm)

    @staticmethod
    def calculate_mu(mu_norm, sigma_norm):
        return np.log(mu_norm / np.sqrt(1 + sigma_norm ** 2 / mu_norm ** 2))
        # return np.log(mu_norm**2 * np.sqrt(1 / (sigma_norm**2 + mu_norm**2)))

    @property
    def sigma(self):
        return self.calculate_sigma(self.mu_norm, self.sigma_norm)

    @staticmethod
    def calculate_sigma(mu_norm, sigma_norm):
        # return np.sqrt(np.log(1 + sigma_norm**2 / mu_norm**2))
        return np.sqrt(np.log(1 + (sigma_norm ** 2 / mu_norm ** 2)))

    def _get_scipy_parameters(self, mu_norm, sigma_norm):
        if (mu_norm is None) != (sigma_norm is None):
            raise RuntimeError(
                "mu_norm and sigma_norm have to be passed both or not at all"
            )

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

    def draw_sample(self, n, mu_norm=None, sigma_norm=None):
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
    
    The parametrization used is the same as described by 
    Haselsteiner et al. (2019) [4]_.  The distributions cumulative distribution 
    function is given by:
    
    :math:`F(x) = \\left[ 1- \\exp \\left(-\\left( \\frac{x}{\\alpha} \\right)^{\\beta} \\right) \\right] ^{\\delta}`
    
    Parameters
    ----------
    alpha : float
        Scale parameter of the exponentiated weibull distribution. Defaults 
        to 1.
    beta : float
        First shape parameter of the exponentiated weibull distribution. 
        Defaults to 1.
    delta : float
        Second shape parameter of the exponentiated weibull distribution. 
        Defaults to 1.
    f_alpha : float
        Fixed alpha parameter of the weibull distribution (e.g. given physical
        parameter). If this parameter is set, alpha is ignored. The fixed 
        parameter does not change, even when fitting the distribution. Defaults to None.
    f_beta : float
       Fixed beta parameter of the weibull distribution (e.g. given physical
       parameter). If this parameter is set, beta is ignored. The fixed 
       parameter does not change, even when fitting the distribution. Defaults to None.
    f_delta : float
        Fixed delta parameter of the weibull distribution (e.g. given physical
        parameter). If this parameter is set, delta is ignored. The fixed 
        parameter does not change, even when fitting the distribution. Defaults to None.

    References
    ----------
    .. [4] Haselsteiner, A.F.; Thoben, K.D. (2019)
        Predicting wave heights for marine design by prioritizing extreme events in
        a global model, Renewable Energy, Volume 156, August 2020, 
        Pages 1146-1157; https://doi.org/10.1016/j.renene.2020.04.112

    """

    @property
    def parameters(self):
        return {"alpha": self.alpha, "beta": self.beta, "delta": self.delta}

    def __init__(
        self, alpha=1, beta=1, delta=1, f_alpha=None, f_beta=None, f_delta=None
    ):
        self.alpha = alpha if f_alpha is None else f_alpha  # scale
        self.beta = beta if f_beta is None else f_beta  # shape
        self.delta = delta if f_delta is None else f_delta  # shape2
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
        return delta, beta, 0, alpha  # shape1, shape2, loc, scale

    def cdf(self, x, alpha=None, beta=None, delta=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        return sts.exponweib.cdf(x, *scipy_par)

    def icdf(self, prob, alpha=None, beta=None, delta=None):
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        return sts.exponweib.ppf(prob, *scipy_par)

    def pdf(self, x, alpha=None, beta=None, delta=None):
        # x = np.asarray(x, dtype=float)  # If x elements are int we cannot use np.nan .
        # This changes x which is unepexted behaviour!
        # x[x<=0] = np.nan  # To avoid warnings with negative and 0-values, use NaN.
        # TODO ensure for all pdf that no nan come up?
        x_greater_zero = np.where(x > 0, x, np.nan)
        scipy_par = self._get_scipy_parameters(alpha, beta, delta)
        _pdf = sts.exponweib.pdf(x_greater_zero, *scipy_par)
        if _pdf.shape == ():  # x was scalar
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
        p0 = {"alpha": self.alpha, "beta": self.beta, "delta": self.delta}

        fparams = {"floc": 0}

        if self.f_delta is not None:
            fparams["f0"] = self.f_delta
        if self.f_beta is not None:
            fparams["f1"] = self.f_beta
        if self.f_alpha is not None:
            fparams["fscale"] = self.f_alpha

        self.delta, self.beta, _, self.alpha = sts.exponweib.fit(
            sample, p0["delta"], p0["beta"], scale=p0["alpha"], **fparams
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
                self.alpha, self.beta = self._estimate_alpha_beta(
                    self.delta, x, p, weights,
                )
            else:
                raise NotImplementedError()
        else:
            delta0 = self.delta
            self.delta = fmin(
                self._wlsq_error, delta0, disp=False, args=(x, p, weights)
            )[0]

            self.alpha, self.beta = self._estimate_alpha_beta(
                self.delta, x, p, weights,
            )

    @staticmethod
    def _estimate_alpha_beta(delta, x, p, w, falpha=None, fbeta=None):

        # As x = 0 causes problems when x_star is calculated, zero-elements
        # are not considered in the parameter estimation.
        indices = np.nonzero(x)
        x = x[indices]
        p = p[indices]
        w = w[indices]

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
        w = w[indices]

        alpha_hat, beta_hat = ExponentiatedWeibullDistribution._estimate_alpha_beta(
            delta, x, p, w
        )

        # Compute the weighted least squares error.
        x_hat = alpha_hat * (-np.log(1 - p ** (1 / delta))) ** (1 / beta_hat)
        wlsq_error = np.sum(w * (x - x_hat) ** 2)

        return wlsq_error


class GeneralizedGammaDistribution(Distribution):
    """
    A 3-parameter generalized Gamma distribution. 
   
    The parametrization is orientated on [5]_ :
    
    :math:`f(x) = \\frac{ \\lambda^{cm} cx^{cm-1} \\exp \\left[ - \\left(\\lambda x^{c} \\right) \\right] }{\\Gamma(m)}`
    
    Parameters
    ----------
    m : float
        First shape parameter of the generalized Gamma distribution. Defaults to 1.
    c : float
        Second shape parameter of the generalized Gamma distribution. Defaults
        to 1.
    lambda\_ : float
        Scale parameter of the generalized Gamma distribution. 
        Defaults to 1.
    f_m : float
        Fixed shape parameter of the generalized Gamma distribution (e.g. 
        given physical parameter). If this parameter is set, m is ignored. The
        fixed parameter does not change, even when fitting the distribution.
        Defaults to None.
    f_c : float
       Fixed second shape parameter of the generalized Gamma distribution (e.g.
       given  physical parameter). If this parameter is set, c is ignored. The
       fixed parameter does not change, even when fitting the distribution.
       Defaults to None. 
    f_lambda\_ : float
        Fixed reciprocal scale parameter of the generalized Gamma distribution 
        (e.g. given physical parameter). If this parameter is set, lambda\_ is 
        ignored. The fixed parameter does not change, even when fitting the distribution.
        Defaults to None.

    References
    ----------
    .. [5] M.K. Ochi, New approach for estimating the severest sea state from 
        statistical data , Coast. Eng. Chapter 38 (1992) 
        pp. 512-525.
        
    """

    def __init__(self, m=1, c=1, lambda_=1, f_m=None, f_c=None, f_lambda_=None):

        self.m = m if f_m is None else f_m  # shape
        self.c = c if f_c is None else f_c  # shape
        self.lambda_ = lambda_ if f_lambda_ is None else f_lambda_  # reciprocal scale
        self.f_m = f_m
        self.f_c = f_c
        self.f_lambda_ = f_lambda_

    @property
    def parameters(self):
        return {"m": self.m, "c": self.c, "lambda_": self.lambda_}

    @property
    def _scale(self):
        return 1 / (self.lambda_)

    @_scale.setter
    def _scale(self, val):
        self.lambda_ = 1 / val

    def _get_scipy_parameters(self, m, c, lambda_):
        if m is None:
            m = self.m
        if c is None:
            c = self.c
        if lambda_ is None:
            scipy_scale = self._scale
        else:
            scipy_scale = 1 / lambda_
        return m, c, 0, scipy_scale  # shape1, shape2, location=0, reciprocal scale

    def cdf(self, x, m=None, c=None, lambda_=None):
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        m : float, optional
            First shape parameter. Defaults to self.m.
        c : float, optional
            The second shape parameter. Defaults to self.c.
        lambda_: float, optional
            The reciprocal scale parameter . Defaults to self.lambda\_.
        
        """

        scipy_par = self._get_scipy_parameters(m, c, lambda_)
        return sts.gengamma.cdf(x, *scipy_par)

    def icdf(self, prob, m=None, c=None, lambda_=None):
        """
        Inverse cumulative distribution function.
        
        Parameters
        ----------
        prob : array_like
            Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        m : float, optional
            First shape parameter. Defaults to self.m.
        c : float, optional
            The second shape parameter. Defaults to self.c.
        lambda_: float, optional
            The reciprocal scale parameter . Defaults to self.lambda\_.
        
        """

        scipy_par = self._get_scipy_parameters(m, c, lambda_)
        return sts.gengamma.ppf(prob, *scipy_par)

    def pdf(self, x, m=None, c=None, lambda_=None):
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the pdf is evaluated.
            Shape: 1-dimensional.
        m : float, optional
            First shape parameter. Defaults to self.m.
        c : float, optional
            The second shape parameter. Defaults to self.k.
        lambda_: float, optional
            The reciprocal scale parameter . Defaults to self.lambda\_.
        
        """

        scipy_par = self._get_scipy_parameters(m, c, lambda_)
        return sts.gengamma.pdf(x, *scipy_par)

    def draw_sample(self, n, m=None, c=None, lambda_=None):
        scipy_par = self._get_scipy_parameters(m, c, lambda_)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.gengamma.rvs(*scipy_par, size=rvs_size)

    def _fit_mle(self, sample):
        p0 = {"m": self.m, "c": self.c, "scale": self._scale}

        fparams = {"floc": 0}

        if self.f_m is not None:
            fparams["fshape1"] = self.f_m
        if self.f_c is not None:
            fparams["fshape2"] = self.f_c
        if self.f_lambda_ is not None:
            fparams["fscale"] = 1 / (self.f_lambda_)

        self.m, self.c, _, self._scale = sts.gengamma.fit(
            sample, p0["m"], p0["c"], scale=p0["scale"], **fparams
        )

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()
 
class VonMisesDistribution(Distribution):
    """
    A Von Mises (Circular Norm) Distribution. 
    
    The distributions probability density function is given by [1]_: 
    
    :math:`f(x) = \\frac{\\exp{\\kappa \\cos{x - \\mu}}}{2 \\pi I_0(k) \\sigma}`
     
    The distribution is used to model wind-wave misalignment in [2]_. 
    Being a circular norm distribution it can be used to model direction. 
    
    Parameters
    ----------
    kappa: float
        Shape parameter
        Defaults to 1.
    mu : float
        Location parameter, the mean.
        Defaults to 0.
    sigma : float
        Scale parameter, the standard deviation.
        Defaults to 1.
        IF NOT FIXED DISTRIBUTION DOES NOT FIT WELL [3]_. The used is
        recommended to set a value for f_sigma (dafaults to 1). 
    f_kappa : float
       Fixed parameter kappa of the von mises distribution (e.g. given 
       physical parameter). If this parameter is set, kappa is ignored. 
       Defaults to None.
    f_mu : float
        Fixed parameter mu of the von mises distribution (e.g. given physical
        parameter). If this parameter is set, mu is ignored. Defaults to 
        None.
    f_sigma : float
       Fixed parameter sigma of the von mises distribution (e.g. given 
       physical parameter). If this parameter is set, sigma is ignored. 
       Defaults to 1.

    
    References
    ----------
    .. [1] Mardia, Kantilal; Jupp, Peter E. (1999). 
       Directional Statistics. Wiley. ISBN 978-0-471-95333-3.
       
    .. [2] Stewart G M, Robertson A, Jonkman J and Lackner M A 2016 
       The creation of a comprehensive metocean data set for offshore 
       wind turbine simulations: Comprehensive metocean data set 
       Wind Energy 19 1151–9
       
       [3] https://stackoverflow.com/questions/39020222/python-scipy-how-to-fit-a-von-mises-distribution
       last accessed: 22/07/2022
    """
    
    def __init__(self, kappa = 1, mu=0, sigma=1, f_kappa=None, f_mu=None, f_sigma=1):

        self.kappa = kappa # shpae
        self.mu = mu  # location
        self.sigma = sigma  # scale
        self.f_kappa = f_kappa
        self.f_mu = f_mu
        self.f_sigma = f_sigma

    @property
    def parameters(self):
        return {"kappa": self.kappa, "mu": self.mu, "sigma": self.sigma}

    def _get_scipy_parameters(self, kappa, mu, sigma):
        if mu is None:
            loc = self.mu
        else:
            loc = mu
        if sigma is None:
            scale = self.sigma
        else:
            scale = self.sigma
        if kappa is None:
            shape = self.kappa
        else: 
            shape = kappa
        return shape, loc, scale  # loc, scale

    def cdf(self, x, kappa = None, mu=None, sigma=None):
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the cdf is evaluated.
            Shape: 1-dimensional.
        kappa : float, optional
            The shape parameter. Defaults to self.kappa .
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(kappa, mu, sigma)
        return sts.vonmises.cdf(x, *scipy_par)

    def icdf(self, prob, kappa= None, mu=None, sigma=None):
        """
        Inverse cumulative distribution function.
        
        Parameters
        ----------
        prob : Probabilities for which the i_cdf is evaluated.
            Shape: 1-dimensional
        kappa : float, optional
            The shape parameter. Defaults to self.kappa .
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(kappa, mu, sigma)
        return sts.vonmises.ppf(prob, *scipy_par)

    def pdf(self, x, kappa=None, mu=None, sigma=None):
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like, 
            Points at which the pdf is evaluated.
            Shape: 1-dimensional.
        kappa : float. optional,
            The shape parameter. Defaults to self.kappa .
        mu : float, optional
            The location parameter. Defaults to self.mu .
        sigma : float, optional
            The scale parameter. Defaults to self.sigma .
        
        """
        scipy_par = self._get_scipy_parameters(kappa, mu, sigma)
        return sts.vonmises.pdf(x, *scipy_par)

    def draw_sample(self, n, kappa=None, mu=None, sigma=None):
        scipy_par = self._get_scipy_parameters(kappa, mu, sigma)
        rvs_size = self._get_rvs_size(n, scipy_par)
        return sts.vonmises.rvs(*scipy_par, size=rvs_size)

    def _fit_mle(self, sample):
        p0 = {"shape": self.kappa, "loc": self.mu, "scale": self.sigma}

        fparams = {}

        if self.f_mu is not None:
            fparams["floc"] = self.f_mu
        if self.f_sigma is not None:
            fparams["fscale"] = self.f_sigma
        if self.f_kappa is not None:
            fparams["fshape"] = self.f_kappa

        self.kappa, self.mu, self.sigma = sts.vonmises.fit(
            sample, p0["shape"], loc=p0["loc"], scale=p0["scale"], **fparams
        )

    def _fit_lsq(self, data, weights):
        raise NotImplementedError()
