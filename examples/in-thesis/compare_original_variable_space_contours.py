import matplotlib.pyplot as plt
import numpy as np

from virocon import (
    read_ec_benchmark_dataset,
    AndContour,
    OrContour,
    DirectSamplingContour,
    HighestDensityContour,
    GlobalHierarchicalModel,
    plot_2D_contour,
    get_OMAE2020_V_Hs
)

# Fit a joint model based on a wind-wave dataset.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
model = GlobalHierarchicalModel(dist_descriptions)
model.fit(data, fit_descriptions)

# Compute the contours.
alpha = 0.1
n = 1000
sample = model.draw_sample(n)
and_contour = AndContour(model, alpha, deg_step=1, sample=sample, allowed_error=0.001)
or_contour = OrContour(model, alpha, deg_step=1, sample=sample, allowed_error=0.001, lowest_theta=2)
direct_sampling = DirectSamplingContour(model, alpha, sample=sample, deg_step=10)
highest_density = HighestDensityContour(model, alpha, limits=[(0, 20), (0, 20)], deltas=0.02)

# Plot the contours.
fig, axs = plt.subplots(2, 2, figsize=[8, 8], sharex=True, sharey=True)
axs[0][0].scatter(
    sample[:, 0],
    sample[:, 1],
    c="k",
    marker=".",
    alpha=0.3,
    rasterized=True,
)
axs[0][0].plot(and_contour.coordinates[0:-1,0], and_contour.coordinates[0:-1,1], c="#BB5566")
axs[0][1].scatter(
    sample[:, 0],
    sample[:, 1],
    c="k",
    marker=".",
    alpha=0.3,
    rasterized=True,
)
axs[0][1].plot(or_contour.coordinates[0:-3,0], or_contour.coordinates[0:-3,1], c="#BB5566")
plot_2D_contour(direct_sampling, sample=sample, ax=axs[1][0])
plot_2D_contour(highest_density, sample=sample, ax=axs[1][1])
titles = [['AND exceedance', 'OR exceedance'],['Angular exceedance', 'Isodensity exceedance']]
for ax2, t2 in zip(axs, titles):
    for ax, t in zip(ax2, t2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  
        ax.set_title(t)
plt.show()
