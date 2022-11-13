"""
Brief example that computes a sea state contour and compares MLE vs WLSQ fitting.
"""
import matplotlib.pyplot as plt
from virocon import (
    read_ec_benchmark_dataset,
    get_OMAE2020_Hs_Tz,
    GlobalHierarchicalModel,
    IFORMContour,
    plot_2D_contour,
)

# Load sea state measurements.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

# Define the structure of the joint distribution model.
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()
model = GlobalHierarchicalModel(dist_descriptions)

# Estimate the model's parameter values with the default method (MLE).
model.fit(data)

# Compute an IFORM contour with a return period of 50 years.
tr = 50  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
contour1 = IFORMContour(model, alpha)

# Estimate the model's parameter values using weighted lesat squares.
fit_description_hs = {"method": "wlsq", "weights": "quadratic"}
my_fit_descriptions = [fit_description_hs, None]
model2 = GlobalHierarchicalModel(dist_descriptions)
model2.fit(data, fit_descriptions=my_fit_descriptions)

# Compute an IFORM contour with a return period of 50 years.
tr = 50  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
contour2 = IFORMContour(model2, alpha)

# Plot the contours.
fig, axs = plt.subplots(1, 2, figsize=[7.5, 4], sharex=True, sharey=True)
plot_2D_contour(contour1, data, semantics=semantics, swap_axis=True, ax=axs[0])
plot_2D_contour(contour2, data, semantics=semantics, swap_axis=True, ax=axs[1])
titles = ["Maximum likelihood estimation", "Weighted least squares"]
for i, (ax, title) in enumerate(zip(axs, titles)):
    ax.set_title(title)
plt.show()
