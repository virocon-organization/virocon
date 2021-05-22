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
    calculate_alpha,
    plot_marginal_quantiles,
    plot_dependence_functions,
    plot_2D_isodensity,
    plot_2D_contour,
)

"""
Use a sea state dataset with the variables Hs and Tz,
fit the join distribution recommended in DNVGL-RP-C203 to 
it and compute an IFORM contour. This example reproduces
the results published in Haseltseiner et al. (2019).

Such a work flow is for example typical in ship design.

Haselsteiner, A. F., Coe, R. G., Manuel, L., Nguyen, P. T. T., 
Martin, N., & Eckert-Gallup, A. (2019). A benchmarking exercise 
on estimating extreme environmental conditions: Methodology & 
baseline results. Proc. 38th International Conference on Ocean, 
Offshore and Arctic Engineering (OMAE 2019). 
https://doi.org/10.1115/OMAE2019-96523

DNV GL. (2017). Recommended practice DNVGL-RP-C205: 
Environmental conditions and environmental loads.
"""

data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")


# A 3-parameter power function (a dependence function).
def _power3(x, a, b, c):
    return a + b * x ** c


# A 3-parameter exponential function (a dependence function).
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)


bounds = [(0, None), (0, None), (None, None)]
power3 = DependenceFunction(_power3, bounds)
exp3 = DependenceFunction(_exp3, bounds)

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
model.fit(data)

axs = plot_marginal_quantiles(model, data)
axs = plot_dependence_functions(model)
ax = plot_2D_isodensity(model, data)

alpha = calculate_alpha(1, 20)
contour = IFORMContour(model, alpha)

coordinates = contour.coordinates
np.testing.assert_allclose(max(coordinates[:, 0]), 5.0, atol=0.5)
np.testing.assert_allclose(max(coordinates[:, 1]), 16.1, atol=0.5)

ax = plot_2D_contour(contour, sample=data)

plt.show()
