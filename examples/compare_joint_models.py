"""
Fit two different  join distribution models to a wind-wave 
dataset and compare their goodness of fits.
"""
import numpy as np

from virocon.plotting import (
    plot_2D_isodensity,
    plot_dependence_functions,
    plot_marginal_quantiles,
)
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    get_DNVGL_Hs_U,
    get_OMAE2020_V_Hs,
    plot_marginal_quantiles,
    plot_2D_isodensity,
)

# Load dataset (this is dataset D from a benchmarking exercise on environmental
# contours, see https://github.com/ec-benchmark-organizers/ec-benchmark
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")

# Define the structure of the first joint distribution model. This model
# is recommended in the DNVGL's "Recommended practice DNVGL-RP-C205: Environmental
# conditions and environmental loads." (2017) in section 3.6.4.
dist_descriptions1, fit_descriptions1, semantics1 = get_DNVGL_Hs_U()
model1 = GlobalHierarchicalModel(dist_descriptions1)

# Define the structure of the first joint distribution model. This model
# was proposed at the OMAE 2020 conference by Haselesteiner et al:
# Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, K.-D. (2020).
# Global hierarchical models for wind and wave contours: Physical interpretations
# of the dependence functions. Proc. 39th International Conference on Ocean,
# Offshore and Arctic Engineering (OMAE 2020). https://doi.org/10.1115/OMAE2020-18668
dist_descriptions2, fit_descriptions2, semantics2 = get_OMAE2020_V_Hs()
model2 = GlobalHierarchicalModel(dist_descriptions2)

# Fit the two models to the data (estimate their parameter values).
# For model 1, we need the variables in the order hs, v (instead of v, hs)
v = data["wind speed (m/s)"].to_numpy()
hs = data["significant wave height (m)"].to_numpy()
hs_v = np.transpose(np.array([hs, v]))
model1.fit(hs_v, fit_descriptions1)
model2.fit(data, fit_descriptions2)

# Analyze the goodness of fit of the marginal distributions with QQ plots.
fig, axs = plt.subplots(1, 2, figsize=[8, 4])
plot_marginal_quantiles(model1, data, semantics=semantics1, axes=axs)
fig.suptitle("DNVGL 2017 model")
fig, axs = plt.subplots(1, 2, figsize=[8, 4])
plot_marginal_quantiles(model2, data, semantics=semantics2, axes=axs)
fig.suptitle("OMAE 2020 model")

# Analyze the overall goodness of fit by plotting isodensity lines.
fig, axs = plt.subplots(1, 2, figsize=[8, 4])
plot_2D_isodensity(model1, hs_v, semantics1, swap_axis=True, ax=axs[0])
plot_2D_isodensity(model2, data, semantics2, ax=axs[1])
axs[0].set_title("DNVGL 2017 model")
axs[1].set_title("OMAE 2020 model")

plt.tight_layout()
plt.show()
