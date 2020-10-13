
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from viroconcom.distributions import (WeibullDistribution, 
                                      LogNormalDistribution,
                                      DependencyFunction, 
                                      ConditionalDistribution)



# %%

# TODO MultivariateModel
class Model(ABC):
    
    @abstractmethod
    def pdf(self, *args, **kwargs):
        pass
    @abstractmethod
    def cdf(self, *args, **kwargs):
        pass
    @abstractmethod
    def ppf(self, *args, **kwargs):
        pass
    @abstractmethod
    def marginal_pdf(self, *args, **kwargs):
        pass
    @abstractmethod        
    def marginal_cdf(self, *args, **kwargs):
        pass   
    @abstractmethod
    def rvs(self, *args, **kwargs):
        pass


class GlobalHierarchicalModel(Model):
    
    def __init__(self, dist_descriptions):
        # TODO check input
        self.distributions = []
        #self.dependencies = []
        self.conditional_on = []
        self.intervall_split_methods = []
        self.dimensions = len(dist_descriptions)
        for dist_desc in dist_descriptions:
            dist_class = dist_desc("distribution")
            if "conditional_on" in dist_desc:
                self.conditional_on.append(dist_desc["conditional_on"])
                #self.dependencies.append(dist_desc["dependency"])
                # TODO check that all parameters are specified in "dependency", else set defaults
                dist = ConditionalDistribution(dist_class, dist_desc["dependency"])
                self.distributions.append(dist)
                self.interval_split_methods.append(self._get_interval_split_method(dist_desc))
            else:
                self.conditional_on.append(None)
                #self.dependencies.append(None)
                self.distributions.append(dist_class())
                self.interval_split_methods.append(None)
                

    @staticmethod
    def _get_interval_split_method(dist_desc):
        if "number_of_intervals" in dist_desc:
            return "number_of_intervals", dist_desc["number_of_intervals"]
        else:
            raise NotImplementedError("Default not set.")
       

    def _split_in_intervals(self, data, dist_idx, conditioning_idx, method, method_n):
        if method == "number_of_intervals":
            return self._split_by_number_of_intervals(data, dist_idx, conditioning_idx, method_n)
        pass # TODO implement
        # check method and call appropriate submethod
        # check output of submethod: are there enough intervals?
        
    @staticmethod 
    def _split_by_number_of_intervals(data, dist_idx, conditioning_idx, n_intervals):
        pass


        
    
    def fit(self, data):
        data = np.array(data)
        
        assert data.shape[-1] == self.dimensions
        
        for i in range(self.dimensions):
            dist = self.distributions[i]
            conditioning_idx = self.conditional_on[i]
            
            if conditioning_idx is None:
                dist.fit(data[:, i])
            else:
                interval_split_method, interval_n = self.interval_split_methods[i] # conditioning_idx statt i
                dist_data, conditioning_data = self._split_in_intervals(data, i, conditioning_idx, interval_split_method, interval_n)
                #dist data  is a list of ndarray 
                # and conditioning_data is a list of interval points
                dist.fit(dist_data, conditioning_data)
    
    
            self.distributions[i] = dist # TODO is the writeback necessary?
                    
                    
    def pdf(self, *args, **kwargs):
        pass
    
    def cdf(self, *args, **kwargs):
        pass
    
    def icdf(self, *args, **kwargs):
        pass
    
    def marginal_pdf(self, *args, **kwargs):
        pass
               
    def marginal_cdf(self, *args, **kwargs):
        pass   
    
    def rvs(self, *args, **kwargs):
        pass
    
    
# %%

# 3-parameter function that asymptotically decreases (a dependence function).
def _asymdecrease3(x, a, b, c):
    return a + b / (1 + c * x)

# Logarithmic square function. Function has two paramters, but 3 are given such
# that in the software all dependence functions can be called with 3 parameters.
def _lnsquare2(x, a, b):
    return np.log(a + b * np.sqrt(np.divide(x, 9.81)))

asymdecrease3 = DependencyFunction(_asymdecrease3)
lnsquare2 = DependencyFunction(_lnsquare2)

dist_description_0 = {"distribution" : WeibullDistribution,
                      }

dist_description_1 = {"distribution" : LogNormalDistribution,
                      "conditional_on" : 0,
                      "dependency" : {"mu": lnsquare2, # TODO dependency umbenennen parameters
                                      "sigma" : asymdecrease3},
                      "number_of_intervals" : 10,
                      }
# TODO number of intervals zu dist_0

ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])