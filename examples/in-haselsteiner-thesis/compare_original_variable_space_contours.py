"""
Compute four types of contour with the same dataset.

Namely a direct sampling contour, a highest density contour, an AND contour
and an OR contour. These contours are all defined in the original variable space.

This figure is presented as Figure 2.8 on page 18 in Haselsteiner, A. F. (2022). 
Offshore structures under extreme loads: A methodology to determine design loads 
[University of Bremen]. https://doi.org/10.26092/elib/1615
"""

import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    AndContour,
    OrContour,
    DirectSamplingContour,
    HighestDensityContour,
    GlobalHierarchicalModel,
    plot_2D_contour,
    get_OMAE2020_V_Hs,
)

# Fit a joint model based on a wind speed - wave height dataset.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
model = GlobalHierarchicalModel(dist_descriptions)
model.fit(data, fit_descriptions)

# Compute the contours.
alpha = 0.1
n = 10000
sample = model.draw_sample(n)
and_contour = AndContour(model, alpha, deg_step=1, sample=sample, allowed_error=0.0001)
or_contour = OrContour(
    model, alpha, deg_step=1, sample=sample, allowed_error=0.0001, lowest_theta=2
)
direct_sampling = DirectSamplingContour(model, alpha, sample=sample, deg_step=10)
highest_density = HighestDensityContour(
    model, alpha, limits=[(0, 20), (0, 20)], deltas=0.02
)

small_sample = model.draw_sample(200)

# Plot the contours.
fig, axs = plt.subplots(2, 2, figsize=[8, 8], sharex=True, sharey=True)
axs[0][0].scatter(
    small_sample[:, 0],
    small_sample[:, 1],
    c="k",
    marker=".",
    alpha=0.3,
    rasterized=True,
)
axs[0][0].plot(
    and_contour.coordinates[0:-1, 0], and_contour.coordinates[0:-1, 1], c="#BB5566"
)
axs[0][1].scatter(
    small_sample[:, 0],
    small_sample[:, 1],
    c="k",
    marker=".",
    alpha=0.3,
    rasterized=True,
)
axs[0][1].plot(
    or_contour.coordinates[0:-3, 0], or_contour.coordinates[0:-3, 1], c="#BB5566"
)
plot_2D_contour(direct_sampling, sample=small_sample, ax=axs[1][0])
plot_2D_contour(highest_density, sample=small_sample, ax=axs[1][1])
titles = [
    ["AND exceedance", "OR exceedance"],
    ["Angular exceedance", "Isodensity exceedance"],
]
for ax2, t2 in zip(axs, titles):
    for ax, t in zip(ax2, t2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(t)
        ax.set_xlim([0, 25])
        ax.set_ylim([0, 7])
        for line in ax.lines:
            line.set_linewidth(2)
            line.set_color("k")

# plt.savefig("bivariate_exceedance.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
