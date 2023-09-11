"""
Computes a sea state contour using Windmeier's EW model (Windmeier, 2022).

This model is defined in Hs-steepness space such that a variable transformation
to Hs-Tz space is necessary.

Windmeier, K.-L. (2022). Modeling the statistical distribution of sea state 
parameters [Master Thesis, University of Bremen]. https://doi.org/10.26092/elib/2181
"""
import matplotlib.pyplot as plt
import pandas as pd

from virocon import (
    read_ec_benchmark_dataset,
    variable_transform,
    get_Windmeier_Hs_S,
    GlobalHierarchicalModel,
    TransformedModel,
    IFORMContour,
    plot_2D_isodensity,
    plot_2D_contour,
)

# Load sea state measurements.
data_hs_tz = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_C_1year.txt")
hs = data_hs_tz["significant wave height (m)"]
tz = data_hs_tz["zero-up-crossing period (s)"]
temp, steepness = variable_transform.hs_tz_to_hs_s(hs, tz)
steepness.name = "steepness"
data_hs_s = pd.concat([hs, steepness], axis=1)

# Define the structure of the joint distribution model.
(
    dist_descriptions,
    fit_descriptions,
    hs_tz_semantics,
    transformations,
) = get_Windmeier_Hs_S()
model = GlobalHierarchicalModel(dist_descriptions)

# Fit the model in Hs-S space.
print("Fitting the model.")
model.fit(data_hs_s, fit_descriptions)

hs_s_semantics = {
    "names": ["Significant wave height", "Steepness"],
    "symbols": ["H_s", "S"],
    "units": ["m", "-"],
}

plot_2D_isodensity(model, data_hs_s, semantics=hs_s_semantics, swap_axis=True)
plt.show()

# Transform the fitted model to Hs-Tz space.
t_model = TransformedModel(
    model,
    transformations["transform"],
    transformations["inverse"],
    transformations["jacobian"],
    precision_factor=0.2,  # Use low precision to speed up test.
    random_state=42,
)

# Compute a contour.
s = (
    "Computing contour. This will take a while because a Monte Carlo "
    "method is used for evaluating the ICDF of the TransFormedModel."
)
print(s)
tr = 1  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
contour = IFORMContour(t_model, alpha, n_points=50)
coords = contour.coordinates


plot_2D_contour(contour, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True)
plot_2D_isodensity(t_model, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True)
plt.show()
