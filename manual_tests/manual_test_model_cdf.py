import numpy as np
import matplotlib.pyplot as plt

from virocon.dependencies import DependenceFunction
from virocon.distributions import (WeibullDistribution,
                                   LogNormalDistribution)
from virocon.models import GlobalHierarchicalModel

  
# %%
# OMEA2019 A
# A 3-parameter power function (a dependence function).
def _power3(x, a=1.47, b=0.214, c=0.641):
    return a + b * x ** c

# A 3-parameter exponential function (a dependence function).
def _exp3(x, a=0.0, b=0.308, c=-0.250):
    return a + b * np.exp(c * x)


power3 = DependenceFunction(_power3)
exp3 = DependenceFunction(_exp3)

dist_description_0 = {"distribution" : WeibullDistribution(lambda_=0.944,
                                                           k=1.48,
                                                           theta=0.0981),
                      }

dist_description_1 = {"distribution" : LogNormalDistribution(),
                      "conditional_on" : 0,
                      "parameters" : {"mu": power3,
                                      "sigma" : exp3},
                      }

ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])

steps = 5
x, dx = np.linspace(1, (10, 15), num=steps, retstep=True)

F_my = ghm.cdf(x)


# %%

import sys 
sys.path.append("../viroconcom")
# sys.path.append("../../viroconcom")
from viroconcom.params import FunctionParam
from viroconcom.distributions import (WeibullDistribution, 
                                      LognormalDistribution, 
                                      MultivariateDistribution)

dist0 = WeibullDistribution(shape=1.48, loc=0.0981, scale=0.944)
dep0 = (None, None, None)

ref_mu = FunctionParam('power3', 1.47, 0.214, 0.641)
ref_sigma = FunctionParam('exp3', 0.0, 0.308, -0.250)

dist1 = LognormalDistribution(sigma=ref_sigma, mu=ref_mu)

dep1 = (0, None, 0)

distributions = [dist0, dist1]
dependencies = [dep0, dep1]

mul_dist = MultivariateDistribution(distributions, dependencies)

F_ref = mul_dist.cdf(x.T, (0,0))


# %%

plt.close("all")
plt.figure()
plt.plot(F_ref, label="ref joint cdf")
plt.plot(F_my, label="my joint cdf")
plt.legend()

np.testing.assert_allclose(F_my, F_ref)





