import numpy as np

from virocon import (GlobalHierarchicalModel, WeibullDistribution, 
                     LogNormalDistribution, ExponentiatedWeibullDistribution,
                     DependenceFunction, WidthOfIntervalSlicer)

def get_DNVGL_Hs_Tz():
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
    
    ghm = GlobalHierarchicalModel([dist_description_hs, dist_description_tz])
    
    return ghm


def get_DNVGL_Hs_U():
    # 3.6.4 DNVGL
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

    ghm = GlobalHierarchicalModel([dist_description_hs, dist_description_u])
    
    return ghm


def get_OMAE2020_Hs_Tz():
    # ref
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

    
    ghm = GlobalHierarchicalModel([dist_description_hs, dist_description_tz])
    
    return ghm
    

def get_OMAE2020_V_Hs():
    #TODO ref

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
    
    
    ghm = GlobalHierarchicalModel([dist_description_v, dist_description_hs])
    
    return ghm
    