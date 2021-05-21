import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# %% load data, prepare common variables

data = pd.read_csv("datasets/OMAE2020_Dataset_D.txt", sep=";")
data.columns = ["Datetime", "V", "Hs"]
data = data[["Hs", "V"]]

x, dx = np.linspace([0.1, 0.1], [6, 22], num=100, retstep=True)

#given_hs = list(range(1, 7))

# %% # vc2
from virocon import GlobalHierarchicalModel
from virocon.predefined import get_DNVGL_Hs_U

dist_descriptions, fit_descriptions, semantics = get_DNVGL_Hs_U()

ghm = GlobalHierarchicalModel(dist_descriptions)
ghm.fit(data, fit_descriptions=fit_descriptions)

# %%
from virocon.plotting import plot_2D_isodensity

plot_2D_isodensity(ghm, data, semantics=semantics)

# %%

my_f = ghm.pdf(x)

my_f_weibull3 = ghm.distributions[0].pdf(x[:, 0])
my_weibull3_params = (ghm.distributions[0].beta, ghm.distributions[0].gamma, ghm.distributions[0].alpha)


my_weibull2 = ghm.distributions[1]
my_given = my_weibull2.conditioning_values
my_f_weibull2 = []
for given in my_given:
    my_f_weibull2.append(my_weibull2.pdf(x[:, 1], given))

my_f_weibull2 = np.stack(my_f_weibull2, axis=1)

my_alphas = np.array([par["alpha"] for par in my_weibull2.parameters_per_interval])
my_betas = np.array([par["beta"] for par in my_weibull2.parameters_per_interval])
my_intervals = my_weibull2.data_intervals

# %% save for automatic test reference

reference_data = {"ref_f_weibull3" : my_f_weibull3, 
                  "ref_weibull3_params" : my_weibull3_params,
                  "ref_givens" : my_given, 
                  "ref_f_weibull2" : my_f_weibull2,
                  "ref_alphas" : my_alphas,
                  "ref_betas" : my_betas
                  }

   
for i, interval in enumerate(my_intervals):
    reference_data[f"ref_interval{i}"] = interval

# np.savez_compressed("reference_data_DNVGL_Hs_V", **reference_data)





# %% viroconcom
import sys 
sys.path.append("../viroconcom")
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour

sample_v = data["V"]
sample_hs = data["Hs"]


# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38 and that is called 'conditonal modeling approach' (CMA).
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_v = {'name': 'Weibull_2p',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('power3', None, 'power3') #Shape, Location, Scale
                      }

# Fit the model to the data.
fit = Fit((sample_hs, sample_v), (dist_description_hs, dist_description_v))


mul_var_dist = fit.mul_var_dist

ref_f = mul_var_dist.pdf(x.T)

ref_f_weibull3 = mul_var_dist.distributions[0].pdf(x[:, 0])

ref_weibull3 = mul_var_dist.distributions[0]
ref_weibull3_params = (ref_weibull3.shape(None), ref_weibull3.loc(None), ref_weibull3.scale(None))

ref_weibull2 = mul_var_dist.distributions[1]
ref_f_weibull2 = []
ref_given = fit.multiple_fit_inspection_data[1].scale_at
#assert all(ref_given == my_given)
for given in ref_given:
    y = np.stack([np.full_like(x[:, 1], given), x[:, 1]])
    ref_f_weibull2.append(ref_weibull2.pdf(x[:, 1], y, (0,  None, 0)))

ref_f_weibull2 = np.stack(ref_f_weibull2, axis=1)

ref_alphas = np.log(fit.multiple_fit_inspection_data[1].scale_value)
ref_betas = np.array(fit.multiple_fit_inspection_data[1].shape_value)
ref_intervals = fit.multiple_fit_inspection_data[1].scale_samples
    
for i in range(len(ref_intervals)):
    assert len(my_intervals) == len(ref_intervals)
    assert sorted(my_intervals[i]) == sorted(ref_intervals[i])
    


# %% debug prints

print(f"Intervals equal: {sorted(my_intervals[i]) == sorted(ref_intervals[i])}")
print(f"sumabs(mu_diff) = {np.sum(np.abs(ref_alphas - my_alphas))}")
print(f"sumabs(sigma_diff) = {np.sum(np.abs(ref_betas - my_betas))}")
print(f"sumabs(f_diff) = {np.sum(np.abs(ref_f - my_f))}")
print(f"sumabs(f_weibull_diff) = {np.sum(np.abs(ref_f_weibull3 - my_f_weibull3))}")
print(f"sumabs(f_ln_diff) = {np.sum(np.abs(ref_f_weibull2 - my_f_weibull2))}")
    
# %% plotting

plt.close("all")
plt.plot(my_f, label="my_f")
plt.plot(ref_f, label="ref_f")
plt.legend()

plt.figure()
plt.plot(x[:, 0], my_f_weibull3, label="my_weibull")
plt.plot(x[:, 0], ref_f_weibull3, label="ref_weibull3")
plt.legend()

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,)
givens = fit.multiple_fit_inspection_data[1].scale_at
j=0
for i in range(len(ref_given))[::2]:
    ax = axes.flatten()[j]
    j += 1
    ax.plot(x[:, 1], my_f_weibull2[:,i], label="my_weibull2")
    ax.plot(x[:, 1], ref_f_weibull2[:, i], label="ref_weibull2")
    ax.hist(fit.multiple_fit_inspection_data[1].scale_samples[i], density=True)
    ax.set_title(f"Hs = {ref_given[i]}")
    ax.legend()

