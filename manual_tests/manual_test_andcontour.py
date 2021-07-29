import matplotlib.pyplot as plt
import pytest
import numpy as np

from virocon import (
    AndContour,
    WeibullDistribution,
    LogNormalDistribution,
    DependenceFunction,
    GlobalHierarchicalModel,
    plot_2D_contour
)

def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x ** c

# A 3-parameter exponential function (a dependence function).
def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
    return a + b * np.exp(c * x)

bounds = [(0, None), (0, None), (None, None)]
power3 = DependenceFunction(_power3, bounds)
exp3 = DependenceFunction(_exp3, bounds)

dist_description_0 = {
    "distribution": WeibullDistribution(alpha=2.776, beta=1.471, gamma=0.8888),
}
dist_description_1 = {
    "distribution": LogNormalDistribution(),
    "conditional_on": 0,
    "parameters": {"mu": power3, "sigma": exp3},
}
model = GlobalHierarchicalModel([dist_description_0, dist_description_1])

alpha = 0.01
n = 10000
sample = model.draw_sample(n)
contour = AndContour(model, alpha, deg_step=1, sample=sample, allowed_error=0.01)
plot_2D_contour(contour, sample=sample, swap_axis=True)
plt.show()
