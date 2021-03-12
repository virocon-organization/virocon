import sys 
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from virocon.dependencies import DependenceFunction
from virocon.distributions import (ExponentiatedWeibullDistribution,
                                   LogNormalDistribution)
from virocon.models import GlobalHierarchicalModel

  
# %%

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

steps = 2

x, dx = np.linspace(0, 20, num=steps, retstep=True)
# x = x[:, np.newaxis]

my_marg_f0 = ghm.marginal_pdf(x, dim=0)
my_marg_f1 = ghm.marginal_pdf(x, dim=1)

my_marg_F0 =  ghm.marginal_cdf(x, dim=0)
my_marg_F1 =  ghm.marginal_cdf(x, dim=1)

p = np.linspace(0.0001, 0.999, num=steps)
my_marg_x0 = ghm.marginal_icdf(p, dim=0)
my_marg_x1 = ghm.marginal_icdf(p, dim=1)

print("my done")


# %%

# import sys 
# sys.path.append("../viroconcom")
sys.path.append("../../viroconcom")
from viroconcom.params import FunctionParam
from viroconcom.distributions import (ExponentiatedWeibullDistribution, 
                                      LognormalDistribution, 
                                      MultivariateDistribution)

dist0 = ExponentiatedWeibullDistribution(shape=0.684, scale=0.207, shape2=7.79)
dep0 = (None, None, None, None)

ref_mu = FunctionParam('lnsquare2', 3.62, 5.77, None)
ref_sigma = FunctionParam('asymdecrease3', 0, 0.324, 0.404)

dist1 = LognormalDistribution(sigma=ref_sigma, mu=ref_mu)

dep1 = (0, None, 0)

distributions = [dist0, dist1]
dependencies = [dep0, dep1]

mul_dist = MultivariateDistribution(distributions, dependencies)

ref_marg_f0 = mul_dist.marginal_pdf(np.squeeze(x), dim=0)
ref_marg_f1 = mul_dist.marginal_pdf(np.squeeze(x), dim=1)

ref_marg_F0 = mul_dist.marginal_cdf(np.squeeze(x), dim=0)
ref_marg_F1 = mul_dist.marginal_cdf(np.squeeze(x), dim=1)

ref_marg_x0 = mul_dist.marginal_icdf(p, dim=0)
ref_marg_x1 = mul_dist.marginal_icdf(p, dim=1)


# %%
plt.close("all")
plt.figure()
plt.plot(x, my_marg_f0, label="my marginal pdf dim=0")
plt.plot(x, ref_marg_f0, label="ref marginal pdf dim=0")
plt.legend()
plt.figure()
plt.plot(x, my_marg_f1, label="my marginal pdf dim=1")
plt.plot(x, ref_marg_f1, label="ref marginal pdf dim=1")
plt.legend()

plt.figure()
plt.plot(x, my_marg_F0, label="my marginal cdf dim=0")
plt.plot(x, ref_marg_F0, label="ref marginal cdf dim=0")
plt.legend()
plt.figure()
plt.plot(x, my_marg_F1, label="my marginal cdf dim=1")
plt.plot(x, ref_marg_F1, label="ref marginal cdf dim=1")
plt.legend()

plt.figure()
plt.plot(p, my_marg_x0, label="my marginal icdf dim=0")
plt.plot(p, ref_marg_x0, label="ref marginal icdf dim=0")
plt.legend()
plt.figure()
plt.plot(p, my_marg_x1, label="my marginal icdf dim=1")
plt.plot(p, ref_marg_x1, label="ref marginal icdf dim=1")
plt.legend()
# g = sns.JointGrid()

np.testing.assert_allclose(my_marg_f0, ref_marg_f0)
np.testing.assert_allclose(my_marg_f1, ref_marg_f1)

np.testing.assert_allclose(my_marg_F0, ref_marg_F0)
np.testing.assert_allclose(my_marg_F1, ref_marg_F1)

np.testing.assert_allclose(my_marg_x0, ref_marg_x0)
np.testing.assert_allclose(my_marg_x1, ref_marg_x1, rtol=0.1)


        
        