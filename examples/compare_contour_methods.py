"""
Use a wind speed - wave height dataset, fit the join distribution
proposed by Haselsteiner et al. (2020) and compare four different 
contour construction methods. 

A 50-year wind speed - wave height contour is for example used in 
offshore wind turbine design.

Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, 
K.-D. (2020). Global hierarchical models for wind and wave contours: 
Physical interpretations of the dependence functions. Proc. 39th 
International Conference on Ocean, Offshore and Arctic Engineering 
(OMAE 2020). https://doi.org/10.1115/OMAE2020-18668
"""

import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    get_OMAE2020_V_Hs,
    IFORMContour,
    ISORMContour,
    DirectSamplingContour,
    HighestDensityContour,
    plot_2D_contour,
)

# Load a wind speed - significant wave height dataset.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")

# Define the structure of the joint distribution model.
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
model = GlobalHierarchicalModel(dist_descriptions)

# Fit the model to the data (estimate the model's parameter values).
model.fit(data, fit_descriptions)

# Compute four types of contours with a return period of 50 years.
state_duration = 1  # hours
return_period = 50  # years
alpha = state_duration / (return_period * 365.25 * 24)
iform = IFORMContour(model, alpha)
isorm = ISORMContour(model, alpha)
direct_sampling = DirectSamplingContour(model, alpha)
highest_density = HighestDensityContour(model, alpha)

# Plot the contours on top of the metocean data.
fig, axs = plt.subplots(4, 1, figsize=[4, 12], sharex=True, sharey=True)
plot_2D_contour(iform, sample=data, semantics=semantics, ax=axs[0])
plot_2D_contour(isorm, sample=data, semantics=semantics, ax=axs[1])
plot_2D_contour(direct_sampling, sample=data, semantics=semantics, ax=axs[2])
plot_2D_contour(highest_density, sample=data, semantics=semantics, ax=axs[3])
titles = ["IFORM", "ISORM", "Direct sampling", "Highest density"]
for i, (ax, title) in enumerate(zip(axs, titles)):
    ax.set_title(title)
    if i < 3:
        ax.set_xlabel("")

plt.tight_layout()
plt.show()
