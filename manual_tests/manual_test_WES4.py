import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from virocon.dependencies import DependenceFunction
from virocon.distributions import WeibullDistribution, LogNormalNormFitDistribution
from virocon.models import GlobalHierarchicalModel
from virocon.contours import IFORMContour, calculate_alpha
from virocon.intervals import WidthOfIntervalSlicer

# https://doi.org/10.5194/wes-4-325-2019

# data = pd.read_pickle("sigma_files_dataframe.pkl", compression="zip")[["U", "sigma_filt_300"]]
# sigma_data_key = "sigma_filt_300"
# data = pd.read_pickle("sigma_files_dataframe.pkl", compression="zip")[["U", sigma_data_key]]
# U = data.U
# sigma = data.sigma_filt_300

# data = pd.read_csv("datasets/sigma_files.csv", index_col="time", parse_dates=True)
# sigma_data_key = "sigma"
# data = data[["U", sigma_data_key]]

# # data = data[data["U"] > 2.5]

# # %% draw subsample from data:
# sample_size = 10000
# rng = np.random.default_rng(42)
# sample_idc = rng.integers(len(data), size=sample_size)
# subsample = data.iloc[sample_idc]
# new_idx = pd.timedelta_range(start="0", periods=sample_size, freq="10min")
# new_idx.name = "time"
# subsample.index = new_idx

# # data = subsample
# print(data.mean())
# print(data.std())
# print(subsample.mean())
# print(subsample.std())

# subsample.to_csv("datasets/WES4_sample.csv")
# plt.figure()
# plt.hist(data["sigma"], density=True, bins=50)
# plt.hist(subsample["sigma"], density=True, bins=50)

# %%

data = pd.read_csv("datasets/WES4_sample.csv", index_col="time")
data.index = pd.to_timedelta(data.index)
sigma_data_key = "sigma"
      
# %%

class MyIntervalSlicer(WidthOfIntervalSlicer):
    
    def _slice(self, data):
        
        interval_slices, interval_centers = super()._slice(data)
        
        #discard slices below 4 m/s
        ok_slices = []
        ok_centers = []
        # ok_slices, ok_centers = zip(*filter(lambda x: x[1] >=4, zip(interval_slices, interval_centers)))
        for slice_, center in zip(interval_slices, interval_centers):
            if center >=4:
                ok_slices.append(slice_)
                ok_centers.append(center)
        
        return ok_slices, ok_centers

# %%
def _poly3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def _poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


poly3 = DependenceFunction(_poly3)
poly2 = DependenceFunction(_poly2)

dim0_description = {"distribution" : WeibullDistribution(),
                    "intervals" : MyIntervalSlicer(width=1, min_number_of_points=5),
                    # "parameters" : {"lambda_" : 9.74,
                    #                 "k" : 2.02,
                    #                 "theta" : 2.2},
                    }

dim1_description = {"distribution" : LogNormalNormFitDistribution(),
                    "conditional_on" : 0,
                    "parameters" : {"mu_norm": poly3,
                                    "sigma_norm" : poly2},
                    }

# shape, loc, scale lognorm?

ghm = GlobalHierarchicalModel([dim0_description, dim1_description])


ghm.fit(data)

# %%

state_duration = 1 / 6
return_period = 50
# alpha = calculate_alpha(state_duration, return_period)

alpha = 1 / (5 * len(data))

# alpha = 10 min / 50 years
# alpha = 10 min / 50 * 365.25 days
# alpha = 10 min / 50 * 365.25 *24 hours
# alpha = 10 min / 50 * 365.25 *24 * 60 min

# alpha = 10 / (50 * 365.25 * 24 * 60)
# alpha = 1E-5

iform = IFORMContour(ghm, alpha)

coordinates = iform.coordinates

# %%

plt.close("all")
plt.figure()
plt.scatter(data["U"], data[sigma_data_key], alpha=0.5, marker=".", c="gray")
contour_x = np.concatenate([coordinates[:, 0], coordinates[0, 0][np.newaxis]])
contour_y = np.concatenate([coordinates[:, 1], coordinates[0, 1][np.newaxis]])
plt.plot(contour_x, contour_y, color="orange")
plt.xlabel("U")
plt.ylabel(r"$\sigma_u$")
plt.xlim(0, 40)
plt.ylim(0, 6)
plt.tight_layout()

# %% 
U_dist = ghm.distributions[0]

x_U = np.linspace(2, 40, num=100)
x_sigma = np.linspace(0.02, 3.6, num=100)

my_f_weib = U_dist.pdf(x_U)

# plt.close("all")
plt.figure()
plt.plot(x_U, my_f_weib, color="orange")
plt.hist(data.U, density=True)
plt.title("Weibull pdf")

my_ln = ghm.distributions[1]
my_givens = my_ln.conditioning_values
my_f_ln = []
for given in my_givens:
    my_f_ln.append(my_ln.pdf(x_sigma, given))

my_f_ln = np.stack(my_f_ln, axis=1)

my_mu_norms = np.array([par["mu_norm"] for par in my_ln.parameters_per_interval])
my_sigma_norms = np.array([par["sigma_norm"] for par in my_ln.parameters_per_interval])
my_intervals = my_ln.data_intervals

# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,)
# givens = my_givens[::3][1:]
# intervals = my_intervals[::3][1:]
# sigma_fs = my_f_ln[::3][1:]
# for i in range(len(givens)):
#     ax = axes.flatten()[i]    
#     ax.plot(x_sigma, sigma_fs[i], label="my_ln")
#     ax.hist(intervals[i], density=True)
#     ax.set_title(f"{int(givens[i]-0.5)}" + "$ \leq U <$" + f"{int(givens[i]+0.5)}"
#                   + " m/s")
#     # ax.legend()
    
# fig.suptitle('LogNormal pdf')
    
i_ref = 0.14


x = np.linspace(data["U"].min(), data["U"].max())
sigma_x = my_ln.conditioning_values
sigma_y = [dist.sigma for dist in my_ln.distributions_per_interval]
mu_x = my_ln.conditioning_values
mu_y = [dist.mu for dist in my_ln.distributions_per_interval]
sigma_norm_dep = my_ln.conditional_parameters["sigma_norm"]
mu_norm_dep = my_ln.conditional_parameters["mu_norm"]
sigma_dep = lambda x : my_ln.distribution_class.calculate_sigma(mu_norm_dep(x), 
                                                                sigma_norm_dep(x))
mu_dep = lambda x: my_ln.distribution_class.calculate_mu(mu_norm_dep(x), 
                                                         sigma_norm_dep(x))
plt.figure()
plt.plot(x, sigma_dep(x), label="sigma dependence function", color="orange")
plt.scatter(sigma_x, sigma_y)
plt.xlabel("U")
plt.title("sigma")
plt.legend()

plt.figure()
plt.plot(x, mu_dep(x), label="mu dependence function", color="orange")
plt.scatter(mu_x, mu_y)
plt.xlabel("U")
plt.title("mu")
plt.legend()

sigma_norm_y = [dist.parameters["sigma_norm"] for dist in my_ln.distributions_per_interval]
plt.figure()
plt.plot(x, sigma_norm_dep(x), label="2nd order poly", color="orange")
plt.scatter(sigma_x, sigma_norm_y)
plt.xlabel("U")
plt.title("sigma_norm")
plt.xlim((0, 35))
plt.ylim((0, 0.6))
plt.legend()

mu_norm_y = [dist.parameters["mu_norm"] for dist in my_ln.distributions_per_interval]
plt.figure()
plt.plot(x, mu_norm_dep(x), label="3rd order poly", color="orange")
plt.scatter(mu_x, mu_norm_y)
plt.xlabel("U")
plt.title("mu_norm")
plt.xlim((0, 35))
plt.ylim((0, 4))
plt.legend()

# %% save data as ref

# reference_data = {"ref_weib_param" : list(U_dist.parameters.values()),
#                   "ref_f_weib" : my_f_weib,
#                   "ref_givens" : my_givens,
#                   "ref_mu_norms" : my_mu_norms,
#                   "ref_sigma_norms" : my_sigma_norms,
#                   "ref_mus" : mu_y,
#                   "ref_sigmas" : sigma_y,
#                   "ref_f_ln" : my_f_ln,
#                   "ref_coordinates" : coordinates,
#                   }

# for i, my_interval in enumerate(my_intervals):
#     reference_data[f"ref_interval{i}"] = my_interval

# np.savez_compressed("reference_data_WES4", **reference_data)
