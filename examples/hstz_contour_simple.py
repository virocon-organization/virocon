"""
Brief example that computes a sea state contour.
"""
import matplotlib.pyplot as plt
from virocon import (read_ec_benchmark_dataset, get_OMAE2020_Hs_Tz, 
    GlobalHierarchicalModel, IFORMContour, plot_2D_contour)

# Load sea state measurements.
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

# Define the structure of the joint distribution model.
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()
model = GlobalHierarchicalModel(dist_descriptions)

# Estimate the model's parameter values (fitting).
model.fit(data, fit_descriptions=fit_descriptions)

# Compute an IFORM contour with a return period of 50 years.
tr = 50 # Return period in years.
ts = 1 # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
contour = IFORMContour(model, alpha)

# Plot the contour.
plot_2D_contour(contour, data, semantics=semantics, swap_axis=True)
plt.show()
