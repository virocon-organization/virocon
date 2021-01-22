
from abc import ABC, abstractmethod

import numpy as np

from virocon.distributions import ConditionalDistribution
from virocon.intervals import NumberOfIntervalsSlicer



# %%


class MultivariateModel(ABC):
    
    @abstractmethod
    def pdf(self, *args, **kwargs):
        pass
    @abstractmethod
    def cdf(self, *args, **kwargs):
        pass
    @abstractmethod
    def icdf(self, *args, **kwargs):
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


class GlobalHierarchicalModel(MultivariateModel):
    
    def __init__(self, dist_descriptions):
        # TODO check input
        self.distributions = []
        #self.dependencies = []
        self.conditional_on = []
        self.interval_slicers = []
        self.dimensions = len(dist_descriptions)
        self.fit_methods = []
        self.fit_weights = []
        for dist_desc in dist_descriptions:
            dist_class = dist_desc["distribution"]
            self.interval_slicers.append(
                dist_desc.get("intervals", 
                              NumberOfIntervalsSlicer(n_intervals=10)))
            if "conditional_on" in dist_desc:
                self.conditional_on.append(dist_desc["conditional_on"])
                #self.dependencies.append(dist_desc["dependency"])
                # TODO check that all parameters are specified in "dependency", else set defaults
                dist = ConditionalDistribution(dist_class, dist_desc["parameters"])
                self.distributions.append(dist)
            else:
                self.conditional_on.append(None)
                #self.dependencies.append(None)
                dist_params = dist_desc.get("parameters")
                if dist_params is not None:
                    self.distributions.append(dist_class(**dist_params))
                else:
                    self.distributions.append(dist_class())
                    
            self.fit_methods.append(dist_desc.get("fit_method"))
            self.fit_weights.append(dist_desc.get("weights"))
                                
        # TODO throw an error if an unknown key is in dist_description   
       
        
    def _split_in_intervals(self, data, dist_idx, conditioning_idx):
        slicer = self.interval_slicers[conditioning_idx]
        conditioning_data = data[:, conditioning_idx]
        interval_slices, interval_centers = slicer.slice_(conditioning_data)
        
        dist_data = [data[int_slice, dist_idx] for int_slice in interval_slices]
        
        return dist_data, interval_centers
        
    
    def fit(self, data):
        data = np.array(data)
        
        assert data.shape[-1] == self.dimensions
        
        for i in range(self.dimensions):
            dist = self.distributions[i]
            conditioning_idx = self.conditional_on[i]
            fit_method = self.fit_methods[i]
            weights = self.fit_weights[i]
            
            if conditioning_idx is None:
                dist.fit(data[:, i], method=fit_method, weights=weights)
            else:
                # interval_split_method, interval_n = self.interval_split_methods[conditioning_idx]
                # min_interval_size = self.min_interval_sizes[conditioning_idx]
                dist_data, conditioning_data = self._split_in_intervals(data, i, 
                                                                        conditioning_idx)
                #dist data  is a list of ndarray 
                # and conditioning_data is a list of interval points
                dist.fit(dist_data, conditioning_data, method=fit_method, weights=weights)
    
    
            self.distributions[i] = dist # TODO is the writeback necessary? -> probably not
                    
                    
    def pdf(self, x):
        if self.conditional_on[0] is not None:
            raise RuntimeError("Illegal state encountered. The first dimension "
                               "has to be independent, but was conditional on "
                               f"{self.conditional_on[0]}.")
            
        x = np.asarray_chkfinite(x)        
        fs = np.empty_like(x)
        
        fs[:, 0] = self.distributions[0].pdf(x[:, 0])
        
        # TODO check that x has the correct size
        for i in range(1, self.dimensions):
            if self.conditional_on[i] is None:
                fs[:, i] = self.distributions[i].pdf(x[:, i])
            else:
                cond_idx = self.conditional_on[i]
                fs[:, i] = np.array([self.distributions[i].pdf(x[j, i], given=x[j, cond_idx]) 
                                     for j in range(len(x))])

        
        return np.prod(fs, axis=-1)
    
    def cdf(self, *args, **kwargs):
        pass
    
    def icdf(self, p):
        if self.conditional_on[0] is not None:
            raise RuntimeError("Illegal state encountered. The first dimension "
                               "has to be independent, but was conditional on "
                               f"{self.conditional_on[0]}.")
            
        p = np.asarray_chkfinite(p)
        x = np.empty_like(p)
        
        x[:, 0] = self.distributions[0].icdf(p[:, 0])
        
        for i in range(1, self.dimensions):
            if self.conditional_on[i] is None:
                x[:, i] = self.distributions[i].icdf(p[:, i])
            else:
                cond_idx = self.conditional_on[i]
                x[:, i] = np.array([self.distributions[i].icdf(p[j, i], given=x[j, cond_idx]) 
                                    for j in range(len(p))])
                
        return x
        
        
    
    def marginal_pdf(self, *args, **kwargs):
        pass
               
    def marginal_cdf(self, *args, **kwargs):
        pass   
    
    def rvs(self, *args, **kwargs):
        pass
    

