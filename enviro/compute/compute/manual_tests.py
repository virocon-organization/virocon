#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:45:29 2017

@author: kai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from .params import ConstantParam, FunctionParam
from .distributions import (WeibullDistribution, LognormalDistribution, NormalDistribution,
                                   KernelDensityDistribution,
                                   MultivariateDistribution)
from .contours import IFormContour, HighestDensityContour
from .fitting import Fit

if __name__ == "__main__":
    #    # %% try fit
#    mul_dist = MultivariateDistribution()
#    import pandas as pd
#
#    test_data = pd.read_csv(
#        os.path.realpath(os.path.join(os.path.dirname(__file__), "..")) + "/testfiles/1yeardata_vanem2012pdf.csv",
#        sep=';',
#        header=None).as_matrix()
#    p = test_data[:, 0].tolist()
#    q = test_data[:, 1].tolist()
#    dist_0 = {'name': 'Weibull', 'dependency': (None, None, None)}
#    dist_1 = {'name': 'Lognormal_2', 'dependency': (0, None, 0), 'functions': ('f2', None, 'f1')}
#    fit = Fit((p, q), (dist_0, dist_1), 10)
#    # calc contour
#    #    beta = 4.35
#    #    n_angles = 40
#    contour = IFormContour(fit.mul_var_dist)
#    # plot contour
#
#    plt.scatter(contour.coordinates[0][0], contour.coordinates[0][1], label="my solution")
#    plt.show()

    # %% try 2d Iform
#
#     #define dependency tuple
#
#    dep1 = (None, None, None)
#    dep2 = (0, None, 0)
#
#    #define parameters
#    shape = ConstantParam(1.471)
#    loc = ConstantParam(0.8888)
#    scale = ConstantParam(2.776)
#    par1 = (shape, loc, scale)
#
#    shape = FunctionParam(0.0400, 0.1748, -0.2243, "exponential")
#    loc = None
#    scale = FunctionParam(0.1, 1.489, 0.1901, "polynomial")
#
#    par2 = (shape, loc, scale)
#
#    del shape, loc, scale
#
#    #create distributions
#    dist1 = WeibullDistribution(*par1)
#    dist2 = LognormalDistribution(*par2)
#
#    distributions = [dist1, dist2,]
#    dependencies = [dep1, dep2,]
#
#    mul_dist = MultivariateDistribution(distributions, dependencies)
#
#    del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
#    #calc contour
#    return_period = 25
#    state_duration = 3
#    n_points = 20
#    contour = IFormContour(mul_dist, return_period, state_duration, n_points=n_points)
#    #plot contour
#    plt.scatter(contour.coordinates[0][0], contour.coordinates[0][1], label="IFORM")
#    plt.show()

# %% try 3d Iform

#    #define dependency tuple
#    dep1 = (None, None, None)
#    dep2 = (0, None, 0)
#    dep3 = (0, None, 0)
#
#
#    #define parameters/home//home/kai/eclipse-workspacekai/eclipse-workspace
#    shape = ConstantParam(1.471)
#    loc = ConstantParam(0.8888)
#    scale = ConstantParam(2.776)
#    par1 = (shape, loc, scale)
#
#    shape = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
#    loc = None
#    scale = FunctionParam(0.1, 1.489, 0.1901, "f1", wrapper=np.exp)
#    par2 = (shape, loc, scale)
#
#    shape = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
#    loc = ConstantParam(0)
#    scale = FunctionParam(0.1, 1.489, 0.1901, "f1", wrapper=np.exp)
#    par3 = (shape, loc, scale)
#
#    del shape, loc, scale
#
#    #create distributions
#    dist1 = WeibullDistribution(*par1)
#    dist2 = WeibullDistribution(*par2)
#    dist3 = WeibullDistribution(*par3)
#
#    distributions = [dist1, dist2, dist3]
#    dependencies = [dep1, dep2, dep3]
#
#    mul_dist = MultivariateDistribution(distributions, dependencies)
#
#    del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
#    #calc contour
#    return_period = 25
#    state_duration = 3
#    n_points = 1000
#    contour = IFormContour(mul_dist, return_period, state_duration, n_angles)
#
#    #plot contour
#    fig = plt.figure()
#    ax2 = fig.add_subplot(111, projection='3d')
#
#    ax2.scatter(contour.coordinates[0][0], contour.coordinates[0][1], contour.coordinates[0][2])
#    plt.show()

# %% try 2d HDC

    #define dependency tuple
    dep1 = (None, None, None)
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

#    del shape, loc, scale

    #create distributions
    dist1 = WeibullDistribution(*par1)
    dist2 = LognormalDistribution(sigma=shape, mu=scale)

    distributions = [dist1, dist2]
    dependencies = [dep1, dep2]

    mul_dist = MultivariateDistribution(distributions, dependencies)

    del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
    #calc contour
    return_period = 50
    state_duration = 3
    limits = [(0, 20), (0, 20)]
    deltas = [0.05, 0.05]

    contour = HighestDensityContour(mul_dist, return_period, state_duration, limits, deltas,)

    #plot contour
    import matplotlib.pyplot as plt
    plt.scatter(contour.coordinates[0][0], contour.coordinates[0][1], label="my solution")

    import pandas as pd
    result = pd.read_csv("testfiles/hdc25.csv")
    plt.scatter(result["Hs"], result["Tz"], label="Andreas solution")
    plt.legend()
    plt.show()

# %% try 3d HDC

#    #define dependency tuple
#    dep1 = (None, None, None)
#    dep2 = (0, None, 0)
#    dep3 = (0, None, 1)
#
#
#    #define parameters/home//home/kai/eclipse-workspacekai/eclipse-workspace
#    shape = ConstantParam(1.471)
#    loc = ConstantParam(0.8888)
#    scale = ConstantParam(2.776)
#    par1 = (shape, loc, scale)
#
#    shape = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
#    loc = None
#    scale = FunctionParam(0.1, 1.489, 0.1901, "f1", wrapper=np.exp)
#    par2 = (shape, loc, scale)
#
#    shape = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
#    loc = ConstantParam(0)
#    scale = FunctionParam(0.1, 1.489, 0.1901, "f1")
#    par3 = (shape, loc, scale)
#
#    del shape, loc, scale
#
#    #create distributions
#    dist1 = WeibullDistribution(*par1)
#    dist2 = LognormalDistribution(*par2)
#    dist3 = WeibullDistribution(*par3)
#
#    distributions = [dist1, dist2, dist3]
#    dependencies = [dep1, dep2, dep3]
#
#    mul_dist = MultivariateDistribution(distributions, dependencies)
#
#    del dist1, dist2, par1, par2, dep1, dep2, dependencies, distributions
#    #calc contour
#    return_period = 50
#    state_duration = 3
#    limits = [(0, 20), (0, 18), (5,10)]
#    deltas = [0.05, 0.05, 0.05]
#    contour = HighestDensityContour(mul_dist, return_period, state_duration, limits, deltas)

# %% test 3D HDC with proper data

# #define dependency tuple
# dep1 = (None, None, None)
# dep2 = (0, None, 0)
# dep3 = (0, None, 0)
#
# #define parameters
# shape = ConstantParam(1.471)
# loc = ConstantParam(0.8888)
# scale = ConstantParam(2.776)
# par1 = (shape, loc, scale)
#
# sigma = FunctionParam(0.0400, 0.1748, -0.2243, "f2")
# mu = FunctionParam(0.1000, 1.489, 0.1901, "f1")
#
# #create distributions
# dist1 = WeibullDistribution(*par1)
# dist2 = LognormalDistribution(sigma=sigma, mu=mu)
# dist3 = LognormalDistribution(sigma=sigma, mu=mu)
#
# distributions = [dist1, dist2, dist3]
# dependencies = [dep1, dep2, dep3]
#
# mul_dist = MultivariateDistribution(distributions, dependencies)
#
# #calc contour
# return_period = 50
# state_duration = 3
# limits = [(0, 20), (0, 20),(0, 20)]
# deltas = [1, 1, 1]
# test_contour_HDC_2 = HighestDensityContour(mul_dist, return_period, state_duration, limits, deltas)
#
#
# result_andi = pd.read_csv("testfiles/hdc50_3dModel.csv")
# fig1 = plt.figure()
# ax2 = fig1.add_subplot(111, projection='3d')
#
# ax2.scatter(result_andi["1"], result_andi["3"], result_andi["2"], c="green")
# plt.show()
#
#
# fig2 = plt.figure()
# ax3 = fig2.add_subplot(111, projection='3d')
# ax3.scatter(test_contour_HDC_2.coordinates[0][0], test_contour_HDC_2.coordinates[0][1],
#             test_contour_HDC_2.coordinates[0][2], c="orange")
# plt.show()



# %% setup test distribution
#    #define the dependency tuples
#    dep1 = (None, None, None)  # independent
#    dep2 = (None, 2, 0)  # loc depends on third, scale depends on first
#    dep3 = (1, 1, None)
#    dep4 = (0, 1, 2)
#
#    #define parameters (shape, loc, scale)
#    shape = ConstantParam(1)
#    loc = ConstantParam(0.5)
#    scale = ConstantParam(2)
#    par1 = (shape, loc , scale)  # (1, 0.5, 2)
#
#    shape = ConstantParam(1)
#    loc = FunctionParam(0, 1, 2, "polynomial")
#    scale = FunctionParam(0, 2, 3, "polynomial")
#    par2 = (shape, loc , scale)  # (1, (x_3)^2, 2(x_1)^3)
#
#    shape = FunctionParam(0.2, 0.5, 2, "exponential")
#    loc = FunctionParam(0, 1, 2, "polynomial")
#    scale = ConstantParam(2)
#    par3 = (shape, loc , scale)  # (0.2+0.5e^(2*x_2), (x_2)^2, 2)
#
#    shape = FunctionParam(0, 1, 1, "polynomial")
#    loc = FunctionParam(0, 1, 1, "polynomial")
#    scale = FunctionParam(0, 1, 1, "polynomial")
#    par4 = (shape, loc , scale)  # (x_1, x_2, x_3)
#
#    #create Distributions
#    dist1 = WeibullDistribution(*par1)
#    dist2 = WeibullDistribution(*par2)
#    dist3 = WeibullDistribution(*par3)
#    dist4 = WeibullDistribution(*par4)
#
#    distributions = [dist1, dist2, dist3, dist4]
#    dependencies = [dep1, dep2, dep3, dep4]
#
#    mul_dist = MultivariateDistribution(distributions, dependencies)
#    #try pickling
#    import pickle
#    depickled_mul_dist = pickle.loads(pickle.dumps(mul_dist))

# %% try cell_averaged_pdf

#    # set the limits for all RV's
#    (lim1, lim2, lim3, lim4) = (7, 15, 5, 8)
#    # set the steps for all RV's
#    (d1, d2, d3, d4) = (0.5, 0.5, 0.5, 0.5)
#
#    # create the sampling points for each RV
#    x1 = np.arange(d1/2, lim1-d1/2, d1)
#    x2 = np.arange(d2/2, lim2-d2/2, d2)
#    x3 = np.arange(d3/2, lim3-d3/2, d3)
#    x4 = np.arange(d4/2, lim4-d4/2, d4)
#
#    #bundle sampling points
#    coords = (x1, x2, x3, x4)
#
#    f = mul_dist.cell_averaged_joint_pdf(coords)
