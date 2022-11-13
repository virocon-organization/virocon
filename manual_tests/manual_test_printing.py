import numpy as np
import pandas as pd


from matplotlib import pyplot as plt


data = pd.read_csv("datasets/OMAE2020_Dataset_D.txt", sep=";")
data.columns = ["Datetime", "V", "Hs"]
data = data[["V", "Hs"]]

from virocon import (
    GlobalHierarchicalModel,
    ExponentiatedWeibullDistribution,
    DependenceFunction,
    WidthOfIntervalSlicer,
)


# # A 4-parameter logististics function (a dependence function).
# def _logistics4(x, a, b, c, d):
#     return a + b / (1 + np.exp(-1 * np.abs(c) * (x - d)))

# A 4-parameter logististics function (a dependence function).
def _logistics4(x, a=1, b=1, c=-1, d=1):
    return a + b / (1 + np.exp(c * (x - d)))


# _logistics4 = lambda x, a, b, c=-1, d=1 : a + b / (1 + np.exp(c * (x - d)))

# A 3-parameter function designed for the scale parameter (alpha) of an
# exponentiated Weibull distribution with shape2=5 (see 'Global hierarchical
# models for wind and wave contours').
def _alpha3(x, a, b, c, d_of_x):
    return (a + b * x ** c) / 2.0445 ** (1 / d_of_x(x))


logistics_bounds = [(0, None), (0, None), (None, 0), (0, None)]

alpha_bounds = [(0, None), (0, None), (None, None)]

beta_dep = DependenceFunction(_logistics4, logistics_bounds, weights=lambda x, y: y)
alpha_dep = DependenceFunction(
    _alpha3, alpha_bounds, d_of_x=beta_dep, weights=lambda x, y: y
)


dist_description_vs = {
    "distribution": ExponentiatedWeibullDistribution(),
    "intervals": WidthOfIntervalSlicer(2, min_n_points=50),
}

dist_description_hs = {
    "distribution": ExponentiatedWeibullDistribution(f_delta=5),
    "conditional_on": 0,
    "parameters": {"alpha": alpha_dep, "beta": beta_dep,},
}


ghm = GlobalHierarchicalModel([dist_description_vs, dist_description_hs])


fit_description_vs = {"method": "wlsq", "weights": "quadratic"}
fit_description_hs = {"method": "wlsq", "weights": "quadratic"}

ghm.fit(data, [fit_description_vs, fit_description_hs])
# %% printing

# print(repr(beta_dep))
# print(repr(alpha_dep))
# print()
# print(ghm.distributions[0])
# print(ghm.distributions[1])
print()
print(ghm)
# print(beta_dep)
# print(alpha_dep)

# %%
# from inspect import getsourcelines
# import re

# func_code = getsourcelines(_logistics4)[0]
# func_head = func_code[0]
# func_head = func_head.replace("def ", "")
# parameter = {"a" : 2, "b": 3, "c" : -4, "d": 1}
# for param_name, param_value in parameter.items():
#     # replace if not last parameter
#     func_head = re.sub(param_name + r"(\=-?\d*)?\s*,",
#                        f"{param_name}={param_value},",
#                        func_head)
#     func_head = re.sub(param_name + r"(\=-?\d*)?\s*\)",
#                        f"{param_name}={param_value})",
#                        func_head)

# s = func_head + "".join(func_code[1:])

# def __str__(self):
#     if isinstance(self.func, partial):
#         func_code = getsourcelines(self.func.func)[0]
#     else:
#         func_code = getsourcelines(self.func)[0]
#     func_head = func_code[0]
#     func_head = func_head.replace("def ", "")
#     for param_name, param_value in self.parameters.items():
#         # replace if not last parameter
#         func_head = re.sub(param_name + r"(\=-?\d*)?\s*,",
#                            f"{param_name}={param_value},",
#                            func_head)
#         func_head = re.sub(param_name + r"(\=-?\d*)?\s*\)",
#                            f"{param_name}={param_value})",
#                            func_head)

#     return func_head + "".join(func_code[1:])
