
from abc import ABC, abstractmethod

import numpy as np

from virocon.distributions import ConditionalDistribution
from virocon.intervals import NumberOfIntervalsSlicer



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
    
    _dist_description_keys = {"distribution", "intervals", "conditional_on",
                              "parameters"}
    
    def __init__(self, dist_descriptions):
        self.distributions = []
        self.conditional_on = []
        self.interval_slicers = []
        self.n_dim = len(dist_descriptions)
        self._check_dist_descriptions(dist_descriptions)
        for dist_desc in dist_descriptions:
            # dist_class = dist_desc["distribution"]
            dist = dist_desc["distribution"]
            self.interval_slicers.append(
                dist_desc.get("intervals", 
                              NumberOfIntervalsSlicer(n_intervals=10)))
            
            if "conditional_on" in dist_desc:
                self.conditional_on.append(dist_desc["conditional_on"])
                dist = ConditionalDistribution(dist, dist_desc["parameters"])
                self.distributions.append(dist)
            else:
                self.conditional_on.append(None)
                self.distributions.append(dist)
                    
            
        if self.conditional_on[0] is not None:
            raise RuntimeError("Illegal state encountered. The first dimension "
                               "has to be independent, but was conditional on "
                               f"{self.conditional_on[0]}.")
                   
                     
    def _check_dist_descriptions(self, dist_descriptions):
        for i, dist_desc in enumerate(dist_descriptions):
            if not "distribution" in dist_desc:
                raise ValueError("Mandatory key 'distribution' missing in "
                                 f"dist_description for dimension {i}")
                
            if "conditional_on" in dist_desc and not "parameters" in dist_desc:
                raise ValueError("For conditional distributions the "
                                 "dist_description key 'parameters' "
                                 f"is mandatory but was missing for dimension {i}.")
                
            unknown_keys = set(dist_desc).difference(self._dist_description_keys)
            if len(unknown_keys) > 0:
                raise ValueError("Unknown key(s) in dist_description for "
                                 f"dimension {i}."
                                 f"Known keys are {self._dist_description_keys}, "
                                 f"but found {unknown_keys}.")
        
        
    def _split_in_intervals(self, data, dist_idx, conditioning_idx):
        slicer = self.interval_slicers[conditioning_idx]
        conditioning_data = data[:, conditioning_idx]
        interval_slices, interval_centers = slicer.slice_(conditioning_data)
        
        dist_data = [data[int_slice, dist_idx] for int_slice in interval_slices]
        
        return dist_data, interval_centers
        
    
    def fit(self, data):
        #data.shape = (n_samples, n_dim)
        data = np.array(data)
        
        if data.shape[-1] != self.n_dim:
            raise ValueError("The dimension of data does not match the "
                             "dimension of the model. "
                             f"The model has {self.n_dim} dimensions, "
                             f"but the data has {data.shape[-1]} dimensions.")
        
        for i in range(self.n_dim):
            dist = self.distributions[i]
            conditioning_idx = self.conditional_on[i]
            
            if conditioning_idx is None:
                dist.fit(data[:, i])
            else:
                dist_data, conditioning_data = self._split_in_intervals(data, i, 
                                                                        conditioning_idx)
                #dist data  is a list of ndarray 
                # and conditioning_data is a list of interval points
                dist.fit(dist_data, conditioning_data)
    
    
            self.distributions[i] = dist # TODO is the writeback necessary? -> probably not
                    
                    
    def pdf(self, x):
        x = np.asarray_chkfinite(x)        
        fs = np.empty_like(x)
        
        fs[:, 0] = self.distributions[0].pdf(x[:, 0])
        
        # TODO check that x has the correct size
        for i in range(1, self.n_dim):
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
        p = np.asarray_chkfinite(p)
        x = np.empty_like(p)
        
        x[:, 0] = self.distributions[0].icdf(p[:, 0])
        
        for i in range(1, self.n_dim):
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
    

