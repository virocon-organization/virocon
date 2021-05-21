import numpy as np
import matplotlib.pyplot as plt

from virocon import (WeibullDistribution, LogNormalDistribution, 
                     GlobalHierarchicalModel, DependenceFunction, 
                     calculate_alpha, HighestDensityContour)



def _power3(x, a=0.1000, b=1.489, c=0.1901):
        return a + b * x ** c
    
# A 3-parameter exponential function (a dependence function).
def _exp3(x, a=0.0400, b=0.1748, c=-0.2243):
    return a + b * np.exp(c * x)

bounds = [(0, None), 
          (0, None), 
          (None, None)]
power3 = DependenceFunction(_power3, bounds)
exp3 = DependenceFunction(_exp3, bounds)

dist_description_0 = {"distribution" : WeibullDistribution(alpha=2.776,
                                                           beta=1.471,
                                                           gamma=0.8888),
                      }
dist_description_1 = {"distribution" : LogNormalDistribution(),
                      "conditional_on" : 0,
                      "parameters" : {"mu": power3,
                                      "sigma" : exp3},
                      }
ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])

alpha = calculate_alpha(3, 50)
limits = [(0, 20), (0, 18)]
deltas = [0.1, 0.1]
my_contour = HighestDensityContour(ghm, alpha, limits, deltas)


my_coordinates = my_contour.coordinates
        
# %%

# np.savez_compressed("reference_data_HDC.npz", ref_coordinates=my_coordinates)
        
# %% viroconcom v1
import sys 
sys.path.append("../viroconcom")      
from viroconcom.distributions import (WeibullDistribution,\
                                        LognormalDistribution,\
                                        MultivariateDistribution)
from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.contours import HighestDensityContour # noqa

#Define dependency tuple
dep1 = (None, None, None)
dep2 = (0, None, 0)
#Define parameters
shape = ConstantParam(1.471)
loc = ConstantParam(0.8888)
scale = ConstantParam(2.776)
par1 = (shape, loc, scale)
mu = FunctionParam('power3', 0.1000, 1.489, 0.1901)
sigma = FunctionParam('exp3', 0.0400, 0.1748, -0.2243)
#Create distributions
dist1 = WeibullDistribution(*par1)
dist2 = LognormalDistribution(mu=mu, sigma=sigma)
distributions = [dist1, dist2]
dependencies = [dep1, dep2]
mul_dist = MultivariateDistribution(distributions, dependencies)
#Calculate contour
n_years = 50
limits = [(0, 20), (0, 18)]
deltas = [0.1, 0.1]
ref_contour = HighestDensityContour(mul_dist, n_years, 3, limits, deltas)

ref_coordinates = np.array(ref_contour.coordinates).T

    
# %% plots
plt.close("all")
x = np.linspace((0, 0), (20, 18), num=100, endpoint=True)
my_f = ghm.pdf(x)
ref_f = mul_dist.pdf(x.T)
assert np.max(ref_f - my_f) == 0

plt.figure()
plt.scatter(*ref_contour.coordinates, marker=".", label="viroconcom v1")
plt.scatter(my_contour.coordinates[:, 0], my_contour.coordinates[:, 1], 
            marker=".", label="virocon v2")
plt.legend()

# np.testing.assert_almost_equal(my_coordinates, ref_coordinates)


    

