
import numpy as np
import matplotlib.pyplot as plt


from virocon import (GlobalHierarchicalModel, ExponentiatedWeibullDistribution, 
                     LogNormalDistribution, DependenceFunction, 
                     calculate_alpha, IFORMContour)


x = np.linspace((0, 0), (10, 10), num=100)

# Logarithmic square function.
def _lnsquare2(x, a=3.62, b=5.77):
    return np.log(a + b * np.sqrt(x / 9.81))

# 3-parameter function that asymptotically decreases (a dependence function).
def _asymdecrease3(x, a=0, b=0.324, c=0.404):
    return a + b / (1 + c * x)


lnsquare2 = DependenceFunction(_lnsquare2)
asymdecrease3 = DependenceFunction(_asymdecrease3)

dist_description_0 = {"distribution" : ExponentiatedWeibullDistribution(alpha=0.207,
                                                                        beta=0.684,
                                                                        delta=7.79),
                      }

dist_description_1 = {"distribution" : LogNormalDistribution(),
                      "conditional_on" : 0,
                      "parameters" : {"mu": lnsquare2,
                                      "sigma" : asymdecrease3},
                      }

ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])


my_f = ghm.pdf(x)

my_f_expweib = ghm.distributions[0].pdf(x[:, 0])
my_expweib_param = (ghm.distributions[0].delta, ghm.distributions[0].beta, ghm.distributions[0].alpha)


my_ln = ghm.distributions[1]
my_given = np.arange(1, 10)
my_f_ln = []
for given in my_given:
    my_f_ln.append(my_ln.pdf(x[:, 1], given))

my_f_ln = np.stack(my_f_ln, axis=1)



state_duration = 3
return_period = 20
alpha = calculate_alpha(state_duration, return_period)
my_iform = IFORMContour(ghm, alpha)

my_coordinates = my_iform.coordinates


# %%

import sys 
sys.path.append("../viroconcom")
from viroconcom.params import FunctionParam
from viroconcom.distributions import (ExponentiatedWeibullDistribution, 
                                      LognormalDistribution, 
                                      MultivariateDistribution)
from viroconcom.contours import IFormContour


dist0 = ExponentiatedWeibullDistribution(shape=0.684, scale=0.207, shape2=7.79)
dep0 = (None, None, None, None)

ref_mu = FunctionParam('lnsquare2', 3.62, 5.77, None)
ref_sigma = FunctionParam('asymdecrease3', 0, 0.324, 0.404)

dist1 = LognormalDistribution(sigma=ref_sigma, mu=ref_mu)

dep1 = (0, None, 0)

distributions = [dist0, dist1]
dependencies = [dep0, dep1]

mul_dist = MultivariateDistribution(distributions, dependencies)


ref_f = mul_dist.pdf(x.T)

ref_f_expweib = mul_dist.distributions[0].pdf(x[:, 0])

ref_ln = mul_dist.distributions[1]
ref_f_ln = []
for given in my_given:
    y = np.stack([np.full_like(x[:, 1], given), x[:, 1]])
    ref_f_ln.append(ref_ln.pdf(x[:, 1], y, (0,  None, 0)))

ref_f_ln = np.stack(ref_f_ln, axis=1)


ref_iform = IFormContour(mul_dist, return_period, state_duration)

ref_coordinates = np.stack(ref_iform.coordinates, axis=1)

# %% 

# np.savez_compressed("reference_data_IFORM.npz", ref_coordinates=ref_coordinates)


# %%

plt.close("all")
plt.plot(my_f, label="my_f")
plt.plot(ref_f, label="ref_f")
plt.legend()

plt.figure()
plt.plot(x[:, 0], my_f_expweib, label="my_expweib")
plt.plot(x[:, 0], ref_f_expweib, label="ref_expweib")
plt.legend()

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True,)
for i in range(len(my_given)):
    ax = axes.flatten()[i]    
    ax.plot(x[:, 1], my_f_ln[:,i], label="my_ln")
    ax.plot(x[:, 1], ref_f_ln[:, i], label="ref_ln")
    ax.set_title(f"Hs = {my_given[i]}")
    ax.legend()


plt.figure()
plt.plot(my_iform.coordinates[:, 1], my_iform.coordinates[:, 0], label="my IFORM")
plt.plot(ref_iform.coordinates[1], ref_iform.coordinates[0], label="ref IFORM")
plt.legend()



np.testing.assert_allclose(my_f_expweib, ref_f_expweib)

np.testing.assert_allclose(my_coordinates, ref_coordinates)


