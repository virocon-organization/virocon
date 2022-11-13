"""
Brief example that computes a wind speed - wave height contour.
"""
import matplotlib.pyplot as plt
from virocon import (
    read_ec_benchmark_dataset,
    get_OMAE2020_V_Hs,
    GlobalHierarchicalModel,
    IFORMContour,
    plot_2D_contour,
)

# Load wind speed - wave height hindcast dataset.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_D.txt")

# Define the structure of the joint distribution model.
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_V_Hs()
model = GlobalHierarchicalModel(dist_descriptions)

# Estimate the model's parameter values (fitting).
model.fit(data, fit_descriptions=fit_descriptions)

# Compute an IFORM contour with a return period of 50 years.
tr = 50  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
contour = IFORMContour(model, alpha)

# Plot the contour.
plot_2D_contour(contour, data, semantics=semantics)
plt.show()
