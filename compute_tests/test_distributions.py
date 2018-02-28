#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:49:33 2017

@author: nb
"""

import unittest


import numpy as np

from enviro.compute.params import ConstantParam, FunctionParam
from enviro.compute.distributions import (WeibullDistribution, LognormalDistribution, NormalDistribution,
                                   KernelDensityDistribution,
                                   MultivariateDistribution)

class MultivariateDistributionTest(unittest.TestCase):
    """
    Create a example MultivariateDistribution
    """
    
    #define dependency tuple
    dep1 = (None, 0, None)
    dep2 = (0, None, 0)

    #define parameters
    shape = ConstantParam(1.471)
    loc = ConstantParam(0.8888)
    scale = ConstantParam(2.776)
    par1 = (shape, loc, scale)

    shape = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
    loc = None
    scale = FunctionParam(0.1, 1.489, 0.1901, "f1")
    par2 = (shape, loc, scale)

    del shape, loc, scale

    #create distributions
    dist1 = WeibullDistribution(*par1)
    dist2 = LognormalDistribution(*par2)

    distributions = [dist1, dist2]
    dependencies = [dep1, dep2]
       
    
    def test_add_distribution_err_msg(self):
        """
        tests if the right exception is raised when distribution1 has a 
        dependency
        """
        
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, self.dependencies)


    def test_add_distribution_iter(self):
        """
        tests if an exception is raised by the function add_distribution when 
        distributions isn't iterable but dependencies is and the other way around
        """
               
        distributions = 1
        with self.assertRaises(ValueError): 
            MultivariateDistribution(distributions, self.dependencies)
        dependencies = 0
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)
        
    def test_add_distribution_length(self):
        """
        tests if an exception is raised when distributions and dependencies 
        are of unequal length
        """
        
        dep3 = (0, None, None)
        dependencies = [self.dep1, self.dep2, dep3]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)
            
    def test_add_distribution_dependencies_length(self):
        """
        tests if an exception is raised when a tuple in dependencies 
        has not length 3
        """
        
        dep1 = (None, None)
        dependencies = [dep1, self.dep2]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)
            
    def test_add_distribution_dependencies_value(self):
        """
        tests if an exception is raised when dependencies has an invalid value
        """
        
        dep1 = (-3, None, None)
        dependencies = [dep1, self.dep2]
        with self.assertRaises(ValueError):
            MultivariateDistribution(self.distributions, dependencies)
            
            
            
    def test_add_distribution_not_iterable(self):
        """
        tests the function when both distributions and dependencies 
        are not iterable
        """
        
        distributions = 1
        dependencies = 2
        with self.assertRaises(ValueError):
            MultivariateDistribution(distributions, dependencies)
            
    
        
class ParametricDistributionTest(unittest.TestCase):
    
    def test_distribution_shape_None(self):
        """
        tests if shape is set to default when it has value 'None'
        """
        
        #define parameters
        shape = None
        loc = ConstantParam(0.8888)
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)
        
        dist = NormalDistribution(*par1)
        shape_test = dist._get_parameter_values(rv_values, dependencies)[0]
        self.assertEqual(shape_test, 1)
        
    
    def test_distribution_loc_None(self):
        """
        tests if loc is set to default when it has value 'None'
        """
        
        #define parameters
        shape = ConstantParam(0.8888)
        loc = None
        scale = ConstantParam(2.776)
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)
        
        dist = WeibullDistribution(*par1)
        loc_test = dist._get_parameter_values(rv_values, dependencies)[1]
        self.assertEqual(loc_test, 0)
        
        
    def test_distribution_loc_scale(self):
        """
        tests if scale is set to default when it has value 'None'
        """
        
        #define parameters
        shape = ConstantParam(0.8888)
        loc = ConstantParam(2.776)
        scale = None
        par1 = (shape, loc, scale)
        rv_values = [0.8, 1, 8]
        dependencies = (0, 1, 1)
        
        dist = NormalDistribution(*par1)
        scale_test = dist._get_parameter_values(rv_values, dependencies)[2]
        self.assertEqual(scale_test, 1)
        
    
    def test_check_parameter_value(self):
        """
        tests if the right exception is raised when the given parameters are 
        not in the valid range of numbers
        """
        
        shape = None
        loc = ConstantParam(0.8888)
        scale = ConstantParam(-2.776)
        par1 = (shape, loc, scale)
       
        dist = WeibullDistribution(*par1)
        
        with self.assertRaises(ValueError):
            dist._check_parameter_value(2, -2.776)
        with self.assertRaises(ValueError):    
            dist._check_parameter_value(2, np.inf)
            
    
if __name__ == '__main__':
    unittest.main()