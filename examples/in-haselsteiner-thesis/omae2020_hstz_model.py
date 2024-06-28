"""
Example of how to use the Hs-Tz model that was proposed in Haselsteiner et al. (2020).

Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, K.-D. (2020).
Global hierarchical models for wind and wave contours:
Physical interpretations of the dependence functions.
Proc. 39th International Conference on Ocean, Offshore and Arctic Engineering (OMAE 2020).
https://doi.org/10.1115/OMAE2020-18668

This figure is presented as the left panel of Figure 6.4 on page 109 in Haselsteiner,
A. F. (2022). Offshore structures under extreme loads: A methodology to determine
design loads [University of Bremen]. https://doi.org/10.26092/elib/1615
"""

import numpy as np
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    ExponentiatedWeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    plot_2D_isodensity,
)

# Load sea state measurements from NDBC buoy 44007.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

# Define the marginal distribution for Hs.
dist_description_hs = {
    "distribution": ExponentiatedWeibullDistribution(),
    "intervals": WidthOfIntervalSlicer(width=0.5, min_n_points=50),
}


def _asymdecrease3(x, a, b, c):
    return a + b / (1 + c * x)


def _lnsquare2(x, a, b, c):
    return np.log(a + b * np.sqrt(np.divide(x, 9.81)))


# Define the conditional distribution for Tz.
bounds = [(0, None), (0, None), (None, None)]
sigma_dep = DependenceFunction(_asymdecrease3, bounds=bounds)
mu_dep = DependenceFunction(_lnsquare2, bounds=bounds)
dist_description_tz = {
    "distribution": LogNormalDistribution(),
    "conditional_on": 0,
    "parameters": {
        "sigma": sigma_dep,
        "mu": mu_dep,
    },
}

# Create the joint model structure.
dist_descriptions = [dist_description_hs, dist_description_tz]
model = GlobalHierarchicalModel(dist_descriptions)

# Define how the model shall be fitted to data and fit it.
fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
fit_descriptions = [fit_description_hs, None]
model.fit(data, fit_descriptions)


# Print the estimated parameter values.
print(model)

# Analyze the model's goodness of fit with an isodensity plot.
semantics = {
    "names": ["Significant wave height", "Zero-up-crossing period"],
    "symbols": ["H_s", "T_z"],
    "units": ["m", "s"],
}
plot_2D_isodensity(model, data, semantics, swap_axis=True)
# plt.savefig("virocon_omae2020_hstz.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
