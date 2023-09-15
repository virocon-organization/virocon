"""
Computes IFORM sea state contours using all predefined Hs-Tz joint model structures.
"""
import matplotlib.pyplot as plt
import pandas as pd

from virocon import (
    read_ec_benchmark_dataset,
    variable_transform,
    get_Windmeier_EW_Hs_S,
    get_Nonzero_EW_Hs_S,
    get_OMAE2020_Hs_Tz,
    get_DNVGL_Hs_Tz,
    GlobalHierarchicalModel,
    TransformedModel,
    IFORMContour,
    plot_2D_contour,
)

# Load sea state measurements.
data_hs_tz = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A_1year.txt")
hs = data_hs_tz["significant wave height (m)"]
tz = data_hs_tz["zero-up-crossing period (s)"]
temp, steepness = variable_transform.hs_tz_to_hs_s(hs, tz)
steepness.name = "steepness"
data_hs_s = pd.concat([hs, steepness], axis=1)

# Define the structure of Windmeier's originial EW model (see DOI: 10.26092/elib/2181).
(
    dist_descriptions,
    fit_descriptions,
    hs_tz_semantics,
    transformations,
) = get_Windmeier_EW_Hs_S()
windmeier_ew_model = GlobalHierarchicalModel(dist_descriptions)

# Fit the model in Hs-S space.
print("Fitting the model.")
windmeier_ew_model.fit(data_hs_s, fit_descriptions)

hs_s_semantics = {
    "names": ["Significant wave height", "Steepness"],
    "symbols": ["H_s", "S"],
    "units": ["m", "-"],
}

# Transform the fitted model to Hs-Tz space.
precision_factor = 0.2  # Use low precision to speed up this example.
random_state = 42
windmeier_t_model = TransformedModel(
    windmeier_ew_model,
    transformations["transform"],
    transformations["inverse"],
    transformations["jacobian"],
    precision_factor=precision_factor,
    random_state=random_state,
)

# Compute a contour.
s = (
    "Computing contours. This will take about 5 minute per contour because a "
    "Monte Carlo method is used for evaluating the ICDF of the TransFormedModel."
)
print(s)
tr = 1  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
n_contour_points = 50
windmeier_model_contour = IFORMContour(
    windmeier_t_model, alpha, n_points=n_contour_points
)
print("1/2 computational expensive contours done.")

# Define the structure of the EW model which has a dependence function for scale
# that evaluates to >0 at Hs=0 (the model was inspired by Windmeier's
# original EW model that was presented in DOI: 10.26092/elib/2181) and
# compute a contour.
(
    dist_descriptions,
    fit_descriptions,
    hs_tz_semantics,
    transformations,
) = get_Nonzero_EW_Hs_S()
ew_model = GlobalHierarchicalModel(dist_descriptions)
ew_model.fit(data_hs_s, fit_descriptions)
ew_t_model = TransformedModel(
    ew_model,
    transformations["transform"],
    transformations["inverse"],
    transformations["jacobian"],
    precision_factor=precision_factor,
    random_state=random_state,
)
ew_model_contour = IFORMContour(ew_t_model, alpha, n_points=n_contour_points)
print("2/2 computational expensive contours done.")

# Compute a contour based on on a DNV model for comparison
(
    dnv_dist_descriptions,
    dnv_fit_descriptions,
    dnv_semantics,
) = get_DNVGL_Hs_Tz()
dnv_model = GlobalHierarchicalModel(dnv_dist_descriptions)
dnv_model.fit(data_hs_tz, fit_descriptions=dnv_fit_descriptions)
dnv_contour = IFORMContour(dnv_model, alpha)

# Compute a contour based on on a OMAE2020 model for comparison
(
    omae2020_dist_descriptions,
    omae2020_fit_descriptions,
    omae2020_semantics,
) = get_OMAE2020_Hs_Tz()
omae2020_model = GlobalHierarchicalModel(omae2020_dist_descriptions)
omae2020_model.fit(data_hs_tz, fit_descriptions=omae2020_fit_descriptions)
omae2020_contour = IFORMContour(omae2020_model, alpha)

# Plot the contours on top of the metocean data.
fig, ax = plt.subplots(1, 1, figsize=[5, 5])
plot_2D_contour(
    dnv_contour, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True, ax=ax
)
plot_2D_contour(
    omae2020_contour, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True, ax=ax
)
plot_2D_contour(
    windmeier_model_contour,
    data_hs_tz,
    semantics=hs_tz_semantics,
    swap_axis=True,
    ax=ax,
)
plot_2D_contour(
    ew_model_contour, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True, ax=ax
)
ax.lines[1].set_color("orchid")
ax.lines[2].set_color("blue")
ax.lines[3].set_color("gray")
ax.legend(
    [ax.lines[0], ax.lines[1], ax.lines[2], ax.lines[3]],
    ["DNV model", "OMAE2020 model", "Windmeier's EW model", "Nonzero EW model"],
    ncol=1,
    frameon=False,
)
ax.set_title(f"{tr}-year IFORM contour")

plt.show()
