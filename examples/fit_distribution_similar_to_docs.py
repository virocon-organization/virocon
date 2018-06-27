from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as sts
from scipy.optimize import curve_fit
from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (WeibullDistribution,\
                                           LognormalDistribution,\
                                           NormalDistribution,\
                                           KernelDensityDistribution,\
                                           MultivariateDistribution)

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
prng = np.random.RandomState(42)
sample_1 = prng.normal(10, 1, 500)
sample_2 = [point + prng.uniform(-5, 5) for point in sample_1]
dist_description_1 = {'name': 'KernelDensity', 'dependency': (None, None, None), 'number_of_intervals': 5}
dist_description_2 = {'name': 'Normal', 'dependency': (None, 0, None), 'functions':(None, 'f1', None)}
my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2), timeout=None)
my_contour = IFormContour(my_fit.mul_var_dist)
#example_plot = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IForm")


# Create a Fit and visualize the result in a HDC contour:
from viroconcom.contours import HighestDensityContour
sample_3 = prng.weibull(2, 500) + 15
sample_4 = [point + prng.uniform(-1, 1) for point in sample_1]
dist_description_1 = {'name': 'Weibull', 'dependency': (None, None, None)}
dist_description_2 = {'name': 'Normal', 'dependency': (None, None, None)}
my_fit = Fit((sample_3, sample_4), (dist_description_1, dist_description_2), timeout=None)
return_period = 50
state_duration = 3
limits = [(0, 20), (0, 20)]
deltas = [0.05, 0.05]
my_contour = HighestDensityContour(my_fit.mul_var_dist, return_period, state_duration, limits, deltas,)
#example_plot2 = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="HDC")


# An Example how to visualize how good your fit is:
dist_description_0 = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 3}
dist_description_1 = {'name': 'Lognormal_1', 'dependency': (None, None, 0), 'functions': (None, None, 'f2')}
my_fit = Fit((sample_1, sample_2), (dist_description_0, dist_description_1), timeout=None)
fig = plt.figure(figsize=(10, 8))
example_text = fig.suptitle("Dependence of 'scale'")
ax_1 = fig.add_subplot(221)
title1 = ax_1.set_title("Fitted curve")
param_grid = my_fit.multiple_fit_inspection_data[1].scale_at
x_1 = np.linspace(5, 15, 100)
ax1_plot = ax_1.plot(param_grid, my_fit.multiple_fit_inspection_data[0].scale_value, 'x')
example_plot1 = ax_1.plot(x_1, my_fit.mul_var_dist.distributions[1].scale(x_1))
ax_2 = fig.add_subplot(222)
title2 = ax_2.set_title("Distribution '1'")
ax2_hist = ax_2.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[0], normed=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[0])
ax2_plot = ax_2.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax_3 = fig.add_subplot(223)
title3 = ax_3.set_title("Distribution '2'")
ax3_hist = ax_3.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[1], normed=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[1])
ax3_plot = ax_3.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax_4 = fig.add_subplot(224)
title4 = ax_4.set_title("Distribution '3'")
ax4_hist = ax_4.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[2], normed=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[2])
ax4_plot = ax_4.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
