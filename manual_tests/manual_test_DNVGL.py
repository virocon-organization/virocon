import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# %% load data, prepare common variables

data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]

x, dx = np.linspace([0.1, 0.1], [6, 22], num=100, retstep=True)

#given_hs = list(range(1, 7))

# %% # vc2
from virocon.models import GlobalHierarchicalModel
from virocon.distributions import (WeibullDistribution, 
                                   LogNormalDistribution,
                                   )
from virocon.dependencies import DependenceFunction
from virocon.intervals import WidthOfIntervalSlicer

# A 3-parameter power function (a dependence function).
def _power3(x, a, b, c):
    return a + b * x ** c

# A 3-parameter exponential function (a dependence function).
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)

bounds = [(0, None), 
          (0, None), 
          (None, None)]

power3 = DependenceFunction(_power3, bounds)
exp3 = DependenceFunction(_exp3, bounds)

dist_description_0 = {"distribution" : WeibullDistribution,
                      "intervals" : WidthOfIntervalSlicer(width=0.5, offset=True)
                      }

dist_description_1 = {"distribution" : LogNormalDistribution,
                      "conditional_on" : 0,
                      "parameters" : {"mu": power3,
                                      "sigma" : exp3},
                      }

ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])


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

# %% save for automatic test reference

reference_data = {"ref_f_weibull" : my_f_weibull, 
                  "ref_weibull_params" : my_weibull_params,
                  "ref_givens" : my_given, 
                  "ref_f_lognorm" : my_f_ln,
                  "ref_mus" : my_mus,
                  "ref_sigmas" : my_sigmas
                  }

   
for i, interval in enumerate(my_intervals):
    reference_data[f"ref_interval{i}"] = interval

np.savez_compressed("reference_data_DNVGL", **reference_data)





# %% viroconcom
import sys 
sys.path.append("../viroconcom")
from viroconcom.fitting import Fit

sample_hs = data["Hs"]
sample_tz = data["T"]


# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38. Its a global hierarchical model, based on the Weibull and LN dist.
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_te = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0),
                       #Shape, Location, Scale
                      'functions': ('exp3', None, 'power3'),
                       #Shape, Location, Scale
                       'min_datapoints_for_fit': 50
                      }

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), (dist_description_hs, dist_description_te))

mul_var_dist = fit.mul_var_dist

ref_f = mul_var_dist.pdf(x.T)

ref_f_weibull = mul_var_dist.distributions[0].pdf(x[:, 0])

ref_weibull = mul_var_dist.distributions[0]
ref_weibull_params = (ref_weibull.shape(None), ref_weibull.loc(None), ref_weibull.scale(None))

ref_ln = mul_var_dist.distributions[1]
ref_f_ln = []
ref_given = fit.multiple_fit_inspection_data[1].scale_at
assert all(ref_given == my_given)
for given in ref_given:
    y = np.stack([np.full_like(x[:, 1], given), x[:, 1]])
    ref_f_ln.append(ref_ln.pdf(x[:, 1], y, (0,  None, 0)))

ref_f_ln = np.stack(ref_f_ln, axis=1)

ref_mus = np.log(fit.multiple_fit_inspection_data[1].scale_value)
ref_sigmas = np.array(fit.multiple_fit_inspection_data[1].shape_value)
ref_intervals = fit.multiple_fit_inspection_data[1].scale_samples
    
for i in range(len(ref_intervals)):
    assert len(my_intervals) == len(ref_intervals)
    assert sorted(my_intervals[i]) == sorted(ref_intervals[i])
    
# data is ordered differently -> can this have an effect?

# %% debug prints

print(f"Intervals equal: {sorted(my_intervals[i]) == sorted(ref_intervals[i])}")
print(f"sumabs(mu_diff) = {np.sum(np.abs(ref_mus - my_mus))}")
print(f"sumabs(sigma_diff) = {np.sum(np.abs(ref_sigmas - my_sigmas))}")
print(f"sumabs(f_diff) = {np.sum(np.abs(ref_f - my_f))}")
print(f"sumabs(f_weibull_diff) = {np.sum(np.abs(ref_f_weibull - my_f_weibull))}")
print(f"sumabs(f_ln_diff) = {np.sum(np.abs(ref_f_ln - my_f_ln))}")
    
# %% plotting

plt.close("all")
plt.plot(my_f, label="my_f")
plt.plot(ref_f, label="ref_f")
plt.legend()

plt.figure()
plt.plot(x[:, 0], my_f_weibull, label="my_weibull")
plt.plot(x[:, 0], ref_f_weibull, label="ref_weibull")
plt.legend()

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,)
givens = fit.multiple_fit_inspection_data[1].scale_at
for i in range(len(ref_given)):
    ax = axes.flatten()[i]    
    ax.plot(x[:, 1], my_f_ln[:,i], label="my_ln")
    ax.plot(x[:, 1], ref_f_ln[:, i], label="ref_ln")
    ax.hist(fit.multiple_fit_inspection_data[1].scale_samples[i], density=True)
    ax.set_title(f"Hs = {ref_given[i]}")
    ax.legend()

