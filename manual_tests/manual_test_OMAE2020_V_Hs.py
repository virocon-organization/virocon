import numpy as np
import pandas as pd


from matplotlib import pyplot as plt


# %% load data, prepare common variables

data = pd.read_csv("datasets/OMAE2020_Dataset_D.txt", sep=";")
data.columns = ["Datetime", "V", "Hs"]
data = data[["V", "Hs"]]


x, dx = np.linspace([0.1, 0.1], [30, 12], num=100, retstep=True)

# %% # vc2
from virocon import GlobalHierarchicalModel
from virocon.predefined import get_OMAE2020_V_Hs

dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()

ghm = GlobalHierarchicalModel(dist_descriptions)
ghm.fit(data, fit_descriptions=fit_descriptions)

# %%
from virocon.plotting import plot_2D_isodensity

plot_2D_isodensity(ghm, data, semantics=semantics)

# %%

my_f = ghm.pdf(x)

my_f_expweib0 = ghm.distributions[0].pdf(x[:, 0])
my_expweib0_params = (
    ghm.distributions[0].alpha,
    ghm.distributions[0].beta,
    ghm.distributions[0].delta,
)


my_expweib1 = ghm.distributions[1]
my_givens = my_expweib1.conditioning_values
my_f_expweib1 = []
for given in my_givens:
    my_f_expweib1.append(my_expweib1.pdf(x[:, 1], given))

my_f_expweib1 = np.stack(my_f_expweib1, axis=1)

my_alphas = np.array([par["alpha"] for par in my_expweib1.parameters_per_interval])
my_betas = np.array([par["beta"] for par in my_expweib1.parameters_per_interval])
my_intervals = my_expweib1.data_intervals


# %% viroconcom
import sys

sys.path.append("../viroconcom")
from viroconcom.fitting import Fit

sample_v = data["V"]
sample_hs = data["Hs"]


# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_v = {
    "name": "Weibull_Exp",
    "dependency": (None, None, None, None),
    "width_of_intervals": 2,
}
dist_description_hs = {
    "name": "Weibull_Exp",
    "fixed_parameters": (None, None, None, 5),
    # shape, location, scale, shape2
    "dependency": (0, None, 0, None),
    # shape, location, scale, shape2
    "functions": ("logistics4", None, "alpha3", None),
    # shape, location, scale, shape2
    "min_datapoints_for_fit": 50,
    "do_use_weights_for_dependence_function": True,
}


# Fit the model to the data.
fit = Fit((sample_v, sample_hs), (dist_description_v, dist_description_hs))
# %%

dist0 = fit.mul_var_dist.distributions[0]


mul_var_dist = fit.mul_var_dist

ref_f = mul_var_dist.pdf(x.T)

ref_f_expweib0 = mul_var_dist.distributions[0].pdf(x[:, 0])
ref_expweib0 = mul_var_dist.distributions[0]
ref_expweib0_params = (
    ref_expweib0.scale(None),
    ref_expweib0.shape(None),
    ref_expweib0.shape2(None),
)

ref_expweib1 = mul_var_dist.distributions[1]
ref_f_expweib1 = []
ref_givens = fit.multiple_fit_inspection_data[1].scale_at
assert all(ref_givens == my_givens)
for given in ref_givens:
    y = np.stack([np.full_like(x[:, 1], given), x[:, 1]])
    ref_f_expweib1.append(ref_expweib1.pdf(x[:, 1], y, (0, None, 0, None)))

ref_f_expweib1 = np.stack(ref_f_expweib1, axis=1)

ref_alphas = np.array(fit.multiple_fit_inspection_data[1].scale_value)
ref_betas = np.array(fit.multiple_fit_inspection_data[1].shape_value)
ref_intervals = fit.multiple_fit_inspection_data[1].scale_samples

# %% save reference data

# reference_data = {"ref_expweib0_params" : ref_expweib0_params,
#                   "ref_f_expweib0" : ref_f_expweib0,
#                   "ref_givens" : ref_givens,
#                   "ref_alphas" : ref_alphas,
#                   "ref_betas" : ref_betas,
#                   "ref_f_expweib1" : ref_f_expweib1
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
plt.plot(x[:, 0], my_f_expweib0, label="my_exp_weibull0")
plt.plot(x[:, 0], ref_f_expweib0, label="ref_exp_weibull0")
plt.legend()

fig, axes = plt.subplots(3, 4, sharex=True, sharey=True,)
givens = fit.multiple_fit_inspection_data[1].scale_at
for i in range(len(ref_givens)):
    ax = axes.flatten()[i]
    ax.plot(x[:, 1], my_f_expweib1[:, i], label="my_exp_weibull1")
    ax.plot(x[:, 1], ref_f_expweib1[:, i], label="ref_exp_weibull1")
    ax.hist(fit.multiple_fit_inspection_data[1].scale_samples[i], density=True)
    ax.set_title(f"V = {ref_givens[i]}")
    ax.legend()


# %%
z = np.linspace(data.V.min(), data.V.max())
my_alpha_x = my_expweib1.conditioning_values
my_alpha_dep = my_expweib1.conditional_parameters["alpha"]
plt.figure()
plt.plot(z, my_alpha_dep(z), label="my_alpha_func")
plt.scatter(my_alpha_x, my_alphas, label="my_alpha")

ref_alpha_x = fit.multiple_fit_inspection_data[1].scale_at
ref_alpha_function_param = mul_var_dist.distributions[1].scale
plt.plot(z, ref_alpha_function_param(z), label="ref_alpha_func")
plt.scatter(ref_alpha_x, ref_alphas, label="ref_alpha")
plt.legend()


plt.figure()
my_beta_x = my_expweib1.conditioning_values
my_beta_dep = my_expweib1.conditional_parameters["beta"]
plt.plot(z, my_beta_dep(z), label="my_beta_func")
plt.scatter(my_beta_x, my_betas, label="my_beta")


ref_beta_x = fit.multiple_fit_inspection_data[1].shape_at
ref_beta_function_param = mul_var_dist.distributions[1].shape
plt.plot(z, ref_beta_function_param(z), label="ref_beta_func")
plt.scatter(ref_beta_x, ref_betas, label="ref_beta")
plt.legend()

# %%
np.testing.assert_almost_equal(my_f, ref_f)

np.testing.assert_almost_equal(my_expweib0_params, ref_expweib0_params)
np.testing.assert_almost_equal(my_f_expweib0, ref_f_expweib0)
for my_interval, ref_interval in zip(my_intervals, ref_intervals):
    np.testing.assert_almost_equal(np.sort(my_interval), np.sort(ref_interval))
np.testing.assert_almost_equal(my_givens, ref_givens)
np.testing.assert_almost_equal(my_alphas, ref_alphas)
np.testing.assert_almost_equal(my_betas, ref_betas)
np.testing.assert_almost_equal(my_f_expweib1, ref_f_expweib1)
