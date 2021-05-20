import numpy as np

from virocon import (WeibullDistribution, LogNormalDistribution, 
                     ExponentiatedWeibullDistribution, DependenceFunction, 
                     WidthOfIntervalSlicer)

def get_DNVGL_Hs_Tz():
    """
    Get DNVGL significant wave height and wave period model.
    
    Get the descriptions necessary to create th significant wave height 
    and wave period model as defined in DNVGL [1]_ in section 3.6.3.
    
    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned. 
        Can be passed to fit function of GlobalHierarchicalModel.
    model_description : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
        
    References
    ----------
    .. [1] TODO
    """
    #TODO docstrings with links to literature
    # DNVGL 3.6.3
    def _power3(x, a, b, c):
        return a + b * x ** c
    
    def _exp3(x, a, b, c):
        return a + b * np.exp(c * x)
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    
    power3 = DependenceFunction(_power3, bounds)
    exp3 = DependenceFunction(_exp3, bounds)
    
    dist_description_hs = {"distribution" : WeibullDistribution(),
                          "intervals" : WidthOfIntervalSlicer(width=0.5)
                          }
    
    dist_description_tz = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"mu": power3,
                                          "sigma" : exp3},
                          }
    
    dist_descriptions = [dist_description_hs, dist_description_tz]
    
    fit_descriptions = None
    
    model_description = {"names" : ["Significant wave height", "Zero-crossing wave period"],
                         "symbols" : ["H_s", "T_z"], 
                         "units" : ["m", "s"]
                         }
    
    return dist_descriptions, fit_descriptions, model_description


def get_DNVGL_Hs_U():
    """
    Get DNVGL significant wave height and wind speed model.
    
    Get the descriptions necessary to create the significant wave height 
    and wind speed model as defined in DNVGL [1]_ in section 3.6.4.
    
    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : None
        Default fit is used so None is returned. 
        Can be passed to fit function of GlobalHierarchicalModel.
    model_description : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
        
    References
    ----------
    .. [1] TODO
    """
    def _power3(x, a, b, c):
        return a + b * x ** c
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    
    alpha_dep = DependenceFunction(_power3, bounds=bounds)
    beta_dep = DependenceFunction(_power3, bounds=bounds)
    
    dist_description_hs = {"distribution" : WeibullDistribution(),
                           "intervals" : WidthOfIntervalSlicer(width=0.5,
                                                               min_n_points=20)
                          }
    
    dist_description_u = {"distribution" : WeibullDistribution(f_gamma=0),
                          "conditional_on" : 0,
                          "parameters" : {"alpha" : alpha_dep,
                                          "beta": beta_dep,
                                          },
                          }

    dist_descriptions = [dist_description_hs, dist_description_u]
    
    fit_descriptions = None
    
    model_description = {"names" : ["Significant wave height", "Mean wind speed"],
                         "symbols" : ["H_s", "U"], 
                         "units" : ["m", "m s$^{-1}$"]
                         }
    
    return dist_descriptions, fit_descriptions, model_description


def get_OMAE2020_Hs_Tz():
    """
    Get OMAE2020 significant wave height and wave period model.
    
    Get the descriptions necessary to create the significant wave height 
    and wave period model as described by Haselsteiner et al. [1]_. 
    
    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : list of dict
        List of dictionaries containing the fit description for each dimension.
        Can be passed to fit function of GlobalHierarchicalModel.
    model_description : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
        
    References
    ----------
    .. [1] TODO
    """
    def _asymdecrease3(x, a, b, c):
        return a + b / (1 + c * x)
    
    def _lnsquare2(x, a, b, c):
        return np.log(a + b * np.sqrt(np.divide(x, 9.81)))
    
    bounds = [(0, None), 
              (0, None), 
              (None, None)]
    
    sigma_dep = DependenceFunction(_asymdecrease3, bounds=bounds)
    mu_dep = DependenceFunction(_lnsquare2, bounds=bounds)
    
    
    dist_description_hs = {"distribution" : ExponentiatedWeibullDistribution(),
                           "intervals" : WidthOfIntervalSlicer(width=0.5, 
                                                               min_n_points=50)
                          }
    
    dist_description_tz = {"distribution" : LogNormalDistribution(),
                          "conditional_on" : 0,
                          "parameters" : {"sigma" : sigma_dep,
                                          "mu": mu_dep,
                                          },
                          }

    
    dist_descriptions = [dist_description_hs, dist_description_tz]
    
    fit_description_hs = {"method" : "wlsq", "weights" : "quadratic"}
    fit_descriptions = [fit_description_hs, None]
    
    model_description = {"names" : ["Significant wave height", "Zero-crossing wave period"],
                         "symbols" : ["H_s", "T_z"], 
                         "units" : ["m", "s"]
                         }
    
    return dist_descriptions, fit_descriptions, model_description
    

def get_OMAE2020_V_Hs():
    """
    Get OMAE2020 wind speed and significant wave height model.
    
    Get the descriptions necessary to create the wind speed and 
    significant wave height model as described by Haselsteiner et al. [1]_. 
    
    Returns
    -------
    dist_descriptions : list of dict
        List of dictionaries containing the dist descriptions for each dimension.
        Can be used to create a GlobalHierarchicalModel.
    fit_descriptions : list of dict
        List of dictionaries containing the fit description for each dimension.
        Can be passed to fit function of GlobalHierarchicalModel.
    model_description : dict
        Dictionary with a semantic description of the model.
        Can be passed to plot functions.
        
    References
    ----------
    .. [1] TODO
    """

    def _logistics4(x, a=1, b=1, c=-1, d=1):
        return a + b / (1 + np.exp(c * (x - d)))
    
    def _alpha3(x, a, b, c, d_of_x):
        return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))
    
    logistics_bounds = [(0, None),
                        (0, None),
                        (None, 0),
                        (0, None)]
    
    alpha_bounds = [(0, None), 
                    (0, None), 
                    (None, None)]
    
    beta_dep = DependenceFunction(_logistics4, logistics_bounds,
                                  weights=lambda x, y : y)
    alpha_dep = DependenceFunction(_alpha3, alpha_bounds, d_of_x=beta_dep, 
                                   weights=lambda x, y : y)
    
    
    dist_description_v = {"distribution" : ExponentiatedWeibullDistribution(),
                          "intervals" : WidthOfIntervalSlicer(2, min_n_points=50)
                          }
    
    dist_description_hs = {"distribution" : ExponentiatedWeibullDistribution(f_delta=5),
                           "conditional_on" : 0,
                           "parameters" : {"alpha" : alpha_dep,
                                           "beta": beta_dep,
                                           },
                           }
    
    
    dist_descriptions = [dist_description_v, dist_description_hs]
    
    fit_description_v = {"method" : "wlsq", "weights" : "quadratic"}
    fit_description_hs = {"method" : "wlsq", "weights" : "quadratic"}
    fit_descriptions = [fit_description_v, fit_description_hs]
    
    model_description = {"names" : ["Mean wind speed", "Significant wave height"],
                         "symbols" : ["V", "H_s"], 
                         "units" : ["m s$^{-1}$", "m", ]
                         }
    
    return dist_descriptions, fit_descriptions, model_description
    