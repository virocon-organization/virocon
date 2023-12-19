"""
Computes IFORM sea state contours using all predefined Hs-Tz joint model structures.
"""
import matplotlib.pyplot as plt
import pandas as pd
import copy

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
fname = "datasets/ec-benchmark_dataset_A_1year.txt"
data_hs_tz = read_ec_benchmark_dataset(fname)
hs = data_hs_tz["significant wave height (m)"]
tz = data_hs_tz["zero-up-crossing period (s)"]
temp, steepness = variable_transform.hs_tz_to_hs_s(hs, tz)
steepness.name = "steepness"
data_hs_s = pd.concat([hs, steepness], axis=1)

# Properties for contour calculation.
tr = 1  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
n_contour_points = 50

# Properties for Monte Carlo methods.
precision_factor = 0.2  # Use low precision to speed up this example.
random_state = 42

# Compute a contour based on on a DNV model.
print("Calculating contour 1/4 (with DNVGL model)")
(
    dnv_dist_descriptions,
    dnv_fit_descriptions,
    dnv_semantics,
) = get_DNVGL_Hs_Tz()
dnv_model = GlobalHierarchicalModel(dnv_dist_descriptions)
dnv_model.fit(data_hs_tz, fit_descriptions=dnv_fit_descriptions)
dnv_contour = IFORMContour(dnv_model, alpha)


# Compute a contour based on a OMAE2020 model.
print("Calculating contour 2/4 (with OMAE2020 model)")
(
    omae2020_dist_descriptions,
    omae2020_fit_descriptions,
    omae2020_semantics,
) = get_OMAE2020_Hs_Tz()
omae2020_model = GlobalHierarchicalModel(omae2020_dist_descriptions)
omae2020_model.fit(data_hs_tz, fit_descriptions=omae2020_fit_descriptions)
omae2020_contour = IFORMContour(omae2020_model, alpha)

# Compute a contour based on Windmeier's originial EW model.
# Before contour calculation the model is transformed from Hs-S variable space to Hs-Tz.
# Contour calculation using this transfromed joint distribution is numerically expensive.
# Alternatively, we could calculate the contour in Hs-S variable space and then transfer
# the contour coordinates to Hs-Tz. This would be computationally faster and this approach
# is shown with the next model ('non-zero EW model').
print(
    "Calculating contour 3/4 (with Windmeier's EW model). "
    "Calculating this contour will take about 5 minutes because "
    "a Monte Carlo method is used for evaluating the ICDF of the TransFormedModel."
)
(
    dist_descriptions,
    fit_descriptions,
    hs_tz_semantics,
    transformations,
) = get_Windmeier_EW_Hs_S()
windmeier_ew_model = GlobalHierarchicalModel(dist_descriptions)
windmeier_ew_model.fit(data_hs_s, fit_descriptions)  # Fit the model in Hs-S space.
hs_s_semantics = {
    "names": ["Significant wave height", "Steepness"],
    "symbols": ["H_s", "S"],
    "units": ["m", "-"],
}
windmeier_t_model = TransformedModel(  # Transform the fitted model to Hs-Tz space.
    windmeier_ew_model,
    transformations["transform"],
    transformations["inverse"],
    transformations["jacobian"],
    precision_factor=precision_factor,
    random_state=random_state,
)
windmeier_model_contour = IFORMContour(
    windmeier_t_model, alpha, n_points=n_contour_points
)

# Compute a contour based on the non-zero EW model.
# The contour is calculated in Hs-S variable space and then transformed to Hs-Tz.
print(
    "Calculating contour 4/4 (with non-zero EW model). "
    "Calculating this contour will be fast because we compute the contour in the "
    "original Hs-S variable space and then transform the coordinates to Hs-Tz space."
)
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
ew_model_contour_hs_s = IFORMContour(ew_model, alpha, n_points=n_contour_points)
ew_model_contour_hs_tz = copy.deepcopy(ew_model_contour_hs_s)
ew_model_contour_hs_tz.coordinates = transformations["inverse"](ew_model_contour_hs_tz.coordinates)


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
    ew_model_contour_hs_tz, data_hs_tz, semantics=hs_tz_semantics, swap_axis=True, ax=ax
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
