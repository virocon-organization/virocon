

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
from viroconcom.contours import IFormContour
from viroconcom.fitting import Fit

sample_1 = WeibullDistribution().draw_sample(1000, 1, 1, 1)
sample_2 = NormalDistribution().draw_sample(1000, None, 1, 1)
dist_description_1 = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 5}
dist_description_2 ={'name': 'Normal', 'dependency': (None, None, None)}
my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2))
my_contour = IFormContour(my_fit.mul_var_dist)

example_plot = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IForm")
plt.show()