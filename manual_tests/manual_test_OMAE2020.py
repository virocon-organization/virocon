import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# %% load data, prepare common variables

data = pd.read_csv("datasets/OMAE202_Dataset_D.txt", sep=";")
data.columns = ["Datetime", "V", "Hs"]
data = data[["V", "Hs"]]


x, dx = np.linspace([0.1, 0.1], [30, 12], num=100, retstep=True)

# %% # vc2
from virocon.models import GlobalHierarchicalModel
from virocon.distributions import (ExponentiatedWeibullDistribution,
                                   DependenceFunction,
                                   )


# # A 4-parameter logististics function (a dependence function).
# def _logistics4(x, a, b, c, d):
#     return a + b / (1 + np.exp(-1 * np.abs(c) * (x - d)))

# A 4-parameter logististics function (a dependence function).
def _logistics4(x, a, b, c, d):
    return a + b / (1 + np.exp(c * (x - d)))


# A 3-parameter function designed for the scale parameter (alpha) of an
# exponentiated Weibull distribution with shape2=5 (see 'Global hierarchical
# models for wind and wave contours').
def _alpha3(x, a, b, c, d_of_x):
    return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))


logistics_bounds = [(0, None),
                    (0, None),
                    (None, 0),
                    (0, None)]

alpha_bounds = [(0, None), 
                (0, None), 
                (None, None)]

logistics4 = DependenceFunction(_logistics4, logistics_bounds)
alpha3 = DependenceFunction(_alpha3, alpha_bounds, d_of_x=logistics4)

dist_description_vs = {"distribution" : ExponentiatedWeibullDistribution,
                       "width_of_intervals" : 2,
                       "min_points_per_interval" : 50
                       }

dist_description_hs = {"distribution" : ExponentiatedWeibullDistribution,
                       "conditional_on" : 0,
                       "parameters" : {"alpha": alpha3,
                                       "beta" : logistics4,
                                       "delta" : 5
                                       },
                       }


dist_description_hs = {'name': 'Weibull_Exp',
                       'fixed_parameters': (None, None, None, 5),
                       # shape, location, scale, shape2
                       'dependency': (0, None, 0, None),
                       # shape, location, scale, shape2
                       'functions': ('logistics4', None, 'alpha3', None),
                       # shape, location, scale, shape2
                       'min_datapoints_for_fit': 50,
                       'do_use_weights_for_dependence_function': True}

ghm = GlobalHierarchicalModel([dist_description_vs, dist_description_hs])


ghm.fit(data)

my_f = ghm.pdf(x)

my_f_weibull = ghm.distributions[0].pdf(x[:, 0])
my_weibull_params = (ghm.distributions[0].k, ghm.distributions[0].theta, ghm.distributions[0].lambda_)


my_ln = ghm.distributions[1]
my_given = my_ln.conditioning_values
my_f_ln = []
for given in my_given:
    my_f_ln.append(my_ln.pdf(x[:, 1], given))

my_f_ln = np.stack(my_f_ln, axis=1)

my_mus = np.array([par["mu"] for par in my_ln.parameters_per_interval])
my_sigmas = np.array([par["sigma"] for par in my_ln.parameters_per_interval])
my_intervals = my_ln.data_intervals


# %% viroconcom
import sys 
sys.path.append("../viroconcom")
from viroconcom.fitting import Fit

sample_v = data["V"]
sample_hs = data["Hs"]


# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_v = {'name': 'Weibull_Exp',
                      'dependency': (None, None, None, None),
                      'width_of_intervals': 2}
dist_description_hs = {'name': 'Weibull_Exp',
                       'fixed_parameters': (None, None, None, 5),
                       # shape, location, scale, shape2
                       'dependency': (0, None, 0, None),
                       # shape, location, scale, shape2
                       'functions': ('logistics4', None, 'alpha3', None),
                       # shape, location, scale, shape2
                       'min_datapoints_for_fit': 50,
                       'do_use_weights_for_dependence_function': True}


# Fit the model to the data.
fit = Fit((sample_v, sample_hs),
          (dist_description_v, dist_description_hs))

dist0 = fit.mul_var_dist.distributions[0]







