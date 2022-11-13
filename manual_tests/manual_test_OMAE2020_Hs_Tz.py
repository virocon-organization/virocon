import numpy as np
import pandas as pd


from matplotlib import pyplot as plt


# %% load data, prepare common variables

data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]


x, dx = np.linspace([0.1, 0.1], [30, 12], num=100, retstep=True)

# %% # vc2
from virocon import GlobalHierarchicalModel
from virocon.predefined import get_OMAE2020_Hs_Tz

dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()

ghm = GlobalHierarchicalModel(dist_descriptions)
ghm.fit(data, fit_descriptions=fit_descriptions)

# %%
from virocon.plotting import plot_2D_isodensity

plot_2D_isodensity(ghm, data, semantics=semantics)

# %%

my_f = ghm.pdf(x)

my_f_expweib = ghm.distributions[0].pdf(x[:, 0])
my_expweib_params = (
    ghm.distributions[0].alpha,
    ghm.distributions[0].beta,
    ghm.distributions[0].delta,
)


my_ln = ghm.distributions[1]
my_givens = my_ln.conditioning_values
my_f_ln = []
for given in my_givens:
    my_f_ln.append(my_ln.pdf(x[:, 1], given))

my_f_ln = np.stack(my_f_ln, axis=1)

my_mus = np.array([par["mu"] for par in my_ln.parameters_per_interval])
my_sigmas = np.array([par["sigma"] for par in my_ln.parameters_per_interval])
my_intervals = my_ln.data_intervals


# %% viroconcom
import sys

sys.path.append("../viroconcom")
from viroconcom.fitting import Fit

sample_hs = data["Hs"]
sample_tz = data["T"]

# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_hs = {
    "name": "Weibull_Exp",
    "dependency": (None, None, None, None),
    "width_of_intervals": 0.5,
}
dist_description_tz = {
    "name": "Lognormal_SigmaMu",
    "dependency": (0, None, 0),  # Shape, Location, Scale
    "functions": ("asymdecrease3", None, "lnsquare2"),  # Shape, Location, Scale
    "min_datapoints_for_fit": 50,
}

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), (dist_description_hs, dist_description_tz))

# %%

dist0 = fit.mul_var_dist.distributions[0]


mul_var_dist = fit.mul_var_dist

ref_f = mul_var_dist.pdf(x.T)

ref_f_expweib = mul_var_dist.distributions[0].pdf(x[:, 0])
ref_expweib = mul_var_dist.distributions[0]
ref_expweib_params = (
    ref_expweib.scale(None),
    ref_expweib.shape(None),
    ref_expweib.shape2(None),
)

ref_ln = mul_var_dist.distributions[1]
ref_f_ln = []
ref_givens = fit.multiple_fit_inspection_data[1].scale_at
assert all(ref_givens == my_givens)
for given in ref_givens:
    y = np.stack([np.full_like(x[:, 1], given), x[:, 1]])
    ref_f_ln.append(ref_ln.pdf(x[:, 1], y, (0, None, 0, None)))

ref_f_ln = np.stack(ref_f_ln, axis=1)

ref_mus = np.array(fit.multiple_fit_inspection_data[1].scale_value)
ref_sigmas = np.array(fit.multiple_fit_inspection_data[1].shape_value)
ref_intervals = fit.multiple_fit_inspection_data[1].scale_samples

# %% save reference data

# reference_data = {"ref_expweib_params" : ref_expweib_params,
#                   "ref_f_expweib" : ref_f_expweib,
#                   "ref_givens" : ref_givens,
#                   "ref_mus" : ref_mus,
#                   "ref_sigmas" : ref_sigmas,
#                   "ref_f_ln" : ref_f_ln
#                   }

# for i, ref_interval in enumerate(ref_intervals):
#     reference_data[f"ref_interval{i}"] = ref_interval

# np.savez_compressed("reference_data_OMAE2020", **reference_data)


# %%

plt.close("all")
plt.plot(my_f, label="my_f")
plt.plot(ref_f, label="ref_f")
plt.legend()

plt.figure()
plt.plot(x[:, 0], my_f_expweib, label="my_exp_weibull0")
plt.plot(x[:, 0], ref_f_expweib, label="ref_exp_weibull0")
plt.legend()

fig, axes = plt.subplots(
    3,
    4,
    sharex=True,
    sharey=True,
)
givens = fit.multiple_fit_inspection_data[1].scale_at
for i in range(len(ref_givens)):
    ax = axes.flatten()[i]
    ax.plot(x[:, 1], my_f_ln[:, i], label="my_exp_weibull1")
    ax.plot(x[:, 1], ref_f_ln[:, i], label="ref_exp_weibull1")
    ax.hist(fit.multiple_fit_inspection_data[1].scale_samples[i], density=True)
    ax.set_title(f"V = {ref_givens[i]}")
    ax.legend()


# %%
z = np.linspace(data.Hs.min(), data.Hs.max())
my_mu_x = my_ln.conditioning_values
my_mu_dep = my_ln.conditional_parameters["mu"]
plt.figure()
plt.plot(z, my_mu_dep(z), label="my_mu_func")
plt.scatter(my_mu_x, my_mus, label="my_mu")

ref_mu_x = fit.multiple_fit_inspection_data[1].scale_at
ref_mu_function_param = mul_var_dist.distributions[1].scale
plt.plot(z, np.log(ref_mu_function_param(z)), label="ref_mu_func")
plt.scatter(ref_mu_x, np.log(ref_mus), label="ref_mu")
plt.legend()


plt.figure()
my_sigma_x = my_ln.conditioning_values
my_sigma_dep = my_ln.conditional_parameters["sigma"]
plt.plot(z, my_sigma_dep(z), label="my_sigma_func")
plt.scatter(my_sigma_x, my_sigmas, label="my_sigma")


ref_sigma_x = fit.multiple_fit_inspection_data[1].shape_at
ref_sigma_function_param = mul_var_dist.distributions[1].shape
plt.plot(z, ref_sigma_function_param(z), label="ref_sigma_func")
plt.scatter(ref_sigma_x, ref_sigmas, label="ref_sigma")
plt.legend()

# %%
np.testing.assert_almost_equal(my_f, ref_f)

np.testing.assert_almost_equal(my_expweib_params, ref_expweib_params)
np.testing.assert_almost_equal(my_f_expweib, ref_f_expweib)
for my_interval, ref_interval in zip(my_intervals, ref_intervals):
    np.testing.assert_almost_equal(np.sort(my_interval), np.sort(ref_interval))
np.testing.assert_almost_equal(my_givens, ref_givens)
np.testing.assert_almost_equal(my_mus, np.log(ref_mus))
np.testing.assert_almost_equal(my_sigmas, ref_sigmas)
np.testing.assert_almost_equal(my_f_ln, ref_f_ln)
