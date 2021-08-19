"""
Example of how to use the wind speed - wave height model that was proposed in Haselsteiner et al. (2020).

Haselsteiner, A. F., Sander, A., Ohlendorf, J.-H., & Thoben, K.-D. (2020). 
Global hierarchical models for wind and wave contours:
Physical interpretations of the dependence functions.
Proc. 39th International Conference on Ocean, Offshore and Arctic Engineering (OMAE 2020). 
https://doi.org/10.1115/OMAE2020-18668
"""

import numpy as np
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    ExponentiatedWeibullDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    plot_2D_isodensity,
)

# Load wind speed - wave height measurements from the coastDat-2 hindcast.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")

# Define the marginal distribution for wind speed.
dist_description_v = {
    "distribution": ExponentiatedWeibullDistribution(),
    "intervals": WidthOfIntervalSlicer(2, min_n_points=50),
}


def _logistics4(x, a=1, b=1, c=-1, d=1):
    return a + b / (1 + np.exp(c * (x - d)))


def _alpha3(x, a, b, c, d_of_x):
    return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))


# Define the conditional distribution for Hs.
logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]
alpha_bounds = [(0, None), (0, None), (None, None)]
beta_dep = DependenceFunction(
    _logistics4,
    logistics_bounds,
    weights=lambda x, y: y,
)
alpha_dep = DependenceFunction(
    _alpha3,
    alpha_bounds,
    d_of_x=beta_dep,
    weights=lambda x, y: y,
)
dist_description_hs = {
    "distribution": ExponentiatedWeibullDistribution(f_delta=5),
    "conditional_on": 0,
    "parameters": {
        "alpha": alpha_dep,
        "beta": beta_dep,
    },
}

# Create the joint model structure.
dist_descriptions = [dist_description_v, dist_description_hs]
model = GlobalHierarchicalModel(dist_descriptions)

# Define how the model shall be fitted to data and fit it.
fit_description_v = {"method": "wlsq", "weights": "quadratic"}
fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
fit_descriptions = [fit_description_v, fit_description_hs]
model.fit(data, fit_descriptions)

# Print the estimated parameter values.
print(model)

# Analyze the model's goodness of fit with an isodensity plot.
semantics = {
    "names": ["Wind speed", "Significant wave height"],
    "symbols": ["V", "H_s"],
    "units": [
        "m s$^{-1}$",
        "m",
    ],
}
plot_2D_isodensity(model, data, semantics)
# plt.savefig("virocon_omae2020_vhs.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
