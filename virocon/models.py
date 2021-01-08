
from abc import ABC, abstractmethod

import numpy as np

from virocon.distributions import ConditionalDistribution



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
        self.interval_split_methods = []
        self.min_interval_sizes = []
        self.dimensions = len(dist_descriptions)
        self.fit_methods = []
        self.fit_weights = []
        for dist_desc in dist_descriptions:
            dist_class = dist_desc["distribution"]
            self.interval_split_methods.append(self._get_interval_split_method(dist_desc))
            self.min_interval_sizes.append(dist_desc.get("min_points_per_interval", 20))
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

    @staticmethod
    def _get_interval_split_method(dist_desc):
        if "number_of_intervals" in dist_desc:
            return "number_of_intervals", dist_desc["number_of_intervals"]
        elif "width_of_intervals" in dist_desc:
            return "width_of_intervals", dist_desc["width_of_intervals"]
        elif "points_per_interval" in dist_desc:
            return "points_per_interval", dist_desc["points_per_interval"]
        else: # set default
            return "number_of_intervals", 10
       

    def _split_in_intervals(self, data, dist_idx, conditioning_idx, method, method_n, min_interval_size):
        if method == "number_of_intervals":
            dist_data, interval_centers =  self._split_by_number_of_intervals(data, dist_idx, conditioning_idx, method_n)
        elif method == "width_of_intervals":
            dist_data, interval_centers = self._split_by_width_of_intervals(data, dist_idx, conditioning_idx, method_n)
        elif method == "points_per_interval":
            raise NotImplementedError("points_per_interval not yet implemented")
        else:
            raise NotImplementedError()
                    
        # only use intervals with at least `min_interval_size` points
        ok_intervals = [interval for interval in 
                        zip(dist_data, interval_centers) 
                        if len(interval[0]) >= min_interval_size]
        dist_data, interval_centers = zip(*ok_intervals)
        # TODO check output of submethod: are there enough intervals?
        return dist_data, interval_centers
        
        
    @staticmethod 
    def _split_by_number_of_intervals(data, dist_idx, conditioning_idx, n_intervals):
        # TODO merge with _split_by_width_of_intervals
        conditioning_data = data[:, conditioning_idx]
        interval_starts, interval_width = np.linspace(min(conditioning_data),
                                                      max(conditioning_data),
                                                      num=n_intervals, 
                                                      endpoint=False,
                                                      retstep=True
                                                      )
        interval_centers = interval_starts + 0.5 * interval_width
        interval_masks = [((conditioning_data >= int_start) & 
                           (conditioning_data < int_start + interval_width)) 
                          for int_start in interval_starts]
        
        dist_data = [data[int_mask, dist_idx] for int_mask in interval_masks]
        
        return dist_data, interval_centers
    
    @staticmethod 
    def _split_by_width_of_intervals(data, dist_idx, conditioning_idx, interval_width):
        # TODO implement check, that interval_starts is a proper array
       conditioning_data = data[:, conditioning_idx]
       interval_starts = np.arange(0, max(conditioning_data), interval_width)
       interval_centers = interval_starts + 0.5 * interval_width
       interval_masks = [((conditioning_data >= int_start) & 
                          (conditioning_data < int_start + interval_width)) 
                         for int_start in interval_starts]
       
       dist_data = [data[int_mask, dist_idx] for int_mask in interval_masks]
       
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
                interval_split_method, interval_n = self.interval_split_methods[conditioning_idx]
                min_interval_size = self.min_interval_sizes[conditioning_idx]
                dist_data, conditioning_data = self._split_in_intervals(data, 
                                                                        i, 
                                                                        conditioning_idx, 
                                                                        interval_split_method, 
                                                                        interval_n, 
                                                                        min_interval_size)
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
    

