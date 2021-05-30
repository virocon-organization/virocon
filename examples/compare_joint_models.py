"""
Fit two different  join distribution models to a wind-wave 
dataset and compare their goodness of fits.
"""

from virocon.plotting import plot_2D_isodensity, plot_dependence_functions, plot_marginal_quantiles
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    get_DNVGL_Hs_U,
    get_OMAE2020_V_Hs,
    plot_marginal_quantiles,
    plot_2D_isodensity
)

# Load dataset (this is dataset D from a benchmarking exercise on environmental
# contours, see https://github.com/ec-benchmark-organizers/ec-benchmark
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")

# Define the structure of the first joint distribution model. This model
# is recommended in the DNVGL's "Recommended practice DNVGL-RP-C205: Environmental 
# conditions and environmental loads." (2017).
dist_descriptions1, fit_descriptions1, semantics1 = get_DNVGL_Hs_U()
model1 = GlobalHierarchicalModel(dist_descriptions1)

# Define the structure of the first joint distribution model. This model
# was proposed at the OMAE 2020 conference by Haselesteiner et al:
#Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, K.-D. (2020). 
# Global hierarchical models for wind and wave contours: Physical interpretations 
# of the dependence functions. Proc. 39th International Conference on Ocean, 
# Offshore and Arctic Engineering (OMAE 2020). https://doi.org/10.1115/OMAE2020-18668
dist_descriptions2, fit_descriptions2, semantics2 = get_OMAE2020_V_Hs()
model2 = GlobalHierarchicalModel(dist_descriptions2)

# Fit the two models to the data (estimate their parameter values).
model1.fit(data, fit_descriptions1)
model2.fit(data, fit_descriptions2)

print(model1)

# Analyze the goodness of fit of the marginal distributions with QQ plots.
fig, axs = plt.subplots(2, 2, figsize=[8, 8])
plot_marginal_quantiles(model1, data, semantics=semantics1, axes=axs[0])
plot_marginal_quantiles(model2, data, semantics=semantics2, axes=axs[1])

#titles = ['IFORM', 'ISORM', 'Direct sampling', 'Highest density']
#for ax, title in zip(axs, titles):
#    ax.set_title(title)

plt.tight_layout()
plt.show()
