from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, ISormContour, HighestDensityContour
import matplotlib.pyplot as plt

# Define the multivariate distribution given in the paper by Vanem (2012)
shape = ConstantParam(1.471)
loc = ConstantParam(0.889)
scale = ConstantParam(2.776)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.
my_sigma = FunctionParam(0.040, 0.175, -0.224, "f2")
my_mu = FunctionParam(0.100, 1.489, 0.190, "f1")
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute an IFORM, an ISORM and a highest density contour.
iform_contour = IFormContour(mul_dist, 25, 3, 100)
isorm_contour = ISormContour(mul_dist, 25, 3, 100)
limits = [(0, 20), (0, 20)] # The limits of the computational domain.
deltas = [0.5, 0.1] # The dimensions of the grid cells.
hdens_contour = HighestDensityContour(mul_dist, 25, 3, limits, deltas)

# Plot the three contours.
plt.scatter(iform_contour.coordinates[0][0], iform_contour.coordinates[0][1],
            label="IFORM contour")
plt.scatter(isorm_contour.coordinates[0][0], isorm_contour.coordinates[0][1],
            label="ISORM contour")
plt.scatter(hdens_contour.coordinates[0][0], hdens_contour.coordinates[0][1],
            label="highest density contour")
plt.xlabel('significant wave height [m]')
plt.ylabel('spectral peak period [s]')
plt.legend()
plt.show()
