
import numpy as np

from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

from virocon.fitting import fit_function, fit_constrained_function, convert_bounds_for_curve_fit
from virocon import DependenceFunction

def _linear(x, a, b):
    return x * a + b

def _exp3(x, a=2, b=2, c=2):
    return a + b * np.exp(c * x)

smallest_positive_float = np.nextafter(0, 1)
# 0 < a < inf
# 0 < b < inf
# exp_bounds = [(smallest_positive_float, np.inf), 
#               (smallest_positive_float, np.inf), 
#               (-np.inf, np.inf)]
# exp_bounds = [(0, np.inf), 
#               (0, np.inf), 
#               (-np.inf, np.inf)]
exp_bounds = [(0, None), 
             (0, None), 
             (None, None)]

linear = DependenceFunction(_linear)
# exp3 = DependenceFunction(_exp3)
exp3 = DependenceFunction(_exp3, bounds=exp_bounds)

rng = np.random.RandomState(42)

x = np.linspace(0.1, 10, num=50)
linear_param = (3.6, 6)
y_linear = linear(x, *linear_param) + 5 * rng.normal(scale=1, size=x.shape)
exp_param = (3, 1,  0.5)
y_exp = exp3(x, *exp_param) + 2 * rng.normal(scale=3, size=x.shape)

my_linear_param = fit_function(linear, x, y_linear, (1, 1), "lsq", None, None)
ref_linear_param = curve_fit(linear, x, y_linear, (1, 1))[0]

exp_p0 = tuple(exp3.parameters.values())

my_exp_param = fit_function(exp3, x, y_exp, exp_p0, "lsq", exp3.bounds)
ref_exp_param = curve_fit(exp3, x, y_exp, exp_p0, 
                          bounds=convert_bounds_for_curve_fit(exp3.bounds))[0]

plt.close("all")
plt.figure()
plt.scatter(x, y_linear, marker="x", c="k")
plt.plot(x, linear(x, *linear_param), label="original linear func")
plt.plot(x, linear(x, *my_linear_param), label="my fit", linewidth=3)
plt.plot(x, linear(x, *ref_linear_param), label="scipy curve fit", linestyle="--")
plt.legend()

plt.figure()
plt.scatter(x, y_exp, marker="x", c="k")
plt.plot(x, exp3(x, *exp_param), label="original exp func")
plt.plot(x, exp3(x, *my_exp_param), label="my fit", linewidth=3)
plt.plot(x, exp3(x, *ref_exp_param), label="scipy curve fit", linestyle="--")
plt.legend()

# %%

# x_mu = np.load("x_mu.npy")
# y_mu = np.load("y_mu.npy")

# my_mu = fit_function(exp3, x_mu, y_mu, exp_p0, "lsq", exp3.bounds)
# ref_mu = curve_fit(exp3, x_mu, y_mu, exp_p0, 
#                    bounds=convert_bounds_for_curve_fit(exp3.bounds))[0]

# plt.figure()
# plt.scatter(x_mu, y_mu, marker="x", c="k")
# plt.plot(x_mu, exp3(x_mu, *my_mu), label="my mu", linewidth=3)
# plt.plot(x_mu, exp3(x_mu, *ref_mu), label="ref_mu", linestyle="--")


