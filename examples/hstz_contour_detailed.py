"""
A comprehensive example that shows the whole workflow of
1) Loading data
2) Defining the model structure for a joint distribution
3) Estimating the parameter values of the model ("Fitting")
4) Computing an environmental contour

This example reproduces the results published in Haseltseiner 
et al. (2019). The Hs-Tz join distribution model recommended 
in DNVGL-RP-C203 (2017) is used.

Haselsteiner, A. F., Coe, R. G., Manuel, L., Nguyen, P. T. T., 
Martin, N., & Eckert-Gallup, A. (2019). A benchmarking exercise 
on estimating extreme environmental conditions: Methodology & 
baseline results. Proc. 38th International Conference on Ocean, 
Offshore and Arctic Engineering (OMAE 2019). 
https://doi.org/10.1115/OMAE2019-96523

DNV GL. (2017). Recommended practice DNVGL-RP-C205: 
Environmental conditions and environmental loads.
"""

import numpy as np
import matplotlib.pyplot as plt

from virocon import (
    read_ec_benchmark_dataset,
    GlobalHierarchicalModel,
    WeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
    IFORMContour,
    plot_marginal_quantiles,
    plot_dependence_functions,
    plot_2D_contour,
)

# Load sea state measurements. This dataset has been used
# in a benchmarking exercise, see https://github.com/ec-benchmark-organizers/ec-benchmark
# The dataset was derived from NDBC buoy 44007, https://www.ndbc.noaa.gov/station_page.php?station=44007
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

# Define the structure of the joint model that we will use to describe
# the the environmental data. To define a joint model, we define the
# univariate parametric distributions and the dependence structure.
# The dependence structure is defined using parametric functions.

# A 3-parameter power function, which will be used as a dependence function.
def _power3(x, a, b, c):
    return a + b * x ** c


# A 3-parameter exponential function, which will be used a dependence function.
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)


# Lower and upper interval boundaries for the three parameter values.
bounds = [(0, None), (0, None), (None, None)]

power3 = DependenceFunction(_power3, bounds, latex="$a + b * x^{c}$")
exp3 = DependenceFunction(_exp3, bounds, latex="$a + b * \exp(c * x)$")

dist_description_0 = {
    "distribution": WeibullDistribution(),
    "intervals": WidthOfIntervalSlicer(width=0.5),
}
dist_description_1 = {
    "distribution": LogNormalDistribution(),
    "conditional_on": 0,
    "parameters": {"mu": power3, "sigma": exp3},
}
model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

# Define a dictionary that describes the model.
semantics = {
    "names": ["Significant wave height", "Zero-crossing period"],
    "symbols": ["H_s", "T_z"],
    "units": ["m", "s"],
}

# Fit the model to the data (estimate the model's parameter values).
model.fit(data)

# Print the estimated parameter values
print(model)

# Create plots to inspect the model's goodness-of-fit.
fig1, axs = plt.subplots(1, 2, figsize=[10, 4.8])
plot_marginal_quantiles(model, data, semantics, axes=axs)
fig2, axs = plt.subplots(1, 2, figsize=[10, 4.8])
plot_dependence_functions(model, semantics, axes=axs)

# Compute an IFORM contour with a return period of 20 years.
state_duration = 1  # hours
return_period = 20  # years
alpha = state_duration / (return_period * 365.25 * 24)
contour = IFORMContour(model, alpha)

# Plot the contour on top of a scatter diagram of the metocean data.
ax = plot_2D_contour(contour, sample=data, semantics=semantics, swap_axis=True)

plt.show()
