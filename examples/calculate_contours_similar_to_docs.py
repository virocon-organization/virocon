from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, HighestDensityContour
import matplotlib.pyplot as plt

# Define a Weibull distribution representing significant wave height.
shape = ConstantParam(1.5)
loc = ConstantParam(1)
scale = ConstantParam(3)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.

# Define a Lognormal distribution representing spectral peak period.
my_sigma = FunctionParam(0.05, 0.2, -0.2, "f2")
my_mu = FunctionParam(0.1, 1.5, 0.2, "f1")
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute an IFORM contour with a return period of 25 years, a sea state
# duration of 3 hours and 100 points along the contour.
iform_contour = IFormContour(mul_dist, 25, 3, 100, timeout=None)