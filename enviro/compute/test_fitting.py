#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:34:37 2017

@author: nb
"""

#import unittest
#from compute.fitting import Fit 
#import pandas as pd
#from multiprocessing import Pool
#import numpy as np
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import scipy.stats as sts
#import collections
#import os
#
#from scipy.optimize import curve_fit
#from compute.params import ConstantParam, FunctionParam
#from compute.distributions import WeibullDistribution, LognormalDistribution, NormalDistribution, MultivariateDistribution
#from compute.contours import IFormContour, HighestDensityContour


#class FittingTest(unittest.TestCase):
#    
#
#    test_data = pd.read_csv(
#        os.path.realpath(os.path.join(os.path.dirname(__file__), "..")) + "/testfiles/1yeardata_vanem2012pdf.csv",
#        sep=';',
#        header=None).as_matrix()
#    p = test_data[:, 0].tolist()
#    q = test_data[:, 1].tolist()
#    dist_0 = {'name': 'Weibull', 'dependency': (None, None, None)}
#    dist_1 = {'name': 'Normal', 'dependency': (0, 0, 0),
#              'functions': ('f1', 'f1', 'f1')}
#    
##    
##    sample_1 = np.random.random_sample(5000,)
##    sample_2 = np.random.random_sample(5000,)
##    
##    dist_description_1 = {'name': 'Weibull', 'dependency': (None, None, None)}
##    dist_description_2 = {'name': 'Lognormal_2', 'dependency': (0, None, 0), 
##                          'functions': ('f1', None, 'f2')}
##    my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2), 15)
##    my_contour = IFormContour(my_fit.mul_var_dist)
##    plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IFORM")
##    plt.show()
#
#    sample_1 = np.random.normal(loc=1, scale=0, size=500)
#    sample_2 = np.random.normal(loc=1, scale=0, size=500)
#    dist_description_1 = {'name': 'Normal', 'dependency': (None, None, None)}
#    dist_description_2 = {'name': 'Lognormal_2', 'dependency': (0, None, 0), 
#                          'functions': ('f1', None, 'f2')}
#    my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2), 15)
#    my_contour = HighestDensityContour(my_fit.mul_var_dist)
#    plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="HDC")
#    plt.show()
#    
    
    
    
if __name__ == '__main__':
    unittest.main()