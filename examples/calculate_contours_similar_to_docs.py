import matplotlib.pyplot as plt

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, HighestDensityContour
from viroconcom.plot import plot_contour

# In this example will use the sea state model that was proposed by
# Vanem and Bitner-Gregersen (2012; DOI: 10.1016/j.apor.2012.05.006) and compute
# environmental contours with return periods of 25 years.

# Define a Weibull distribution representing significant wave height.
dist0 = WeibullDistribution(shape=1.471, loc=0.8888, scale=2.776)
dep0 = (None, None, None) # All three parameters are independent.

# Define a Lognormal distribution representing spectral peak period.
my_sigma = FunctionParam('exp3', 0.04, 0.1748, -0.2243)
my_mu = FunctionParam('power3', 0.1, 1.489, 0.1901)
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute an IFORM contour with a return period of 25 years and a sea state
# duration of 6 hours. Calculate 90 points along the contour.
iform_contour = IFormContour(mul_dist, 25, 6, 90)

# Compute a highest density contour with the same settings (25 years return
# period, 6 hour sea state duration).
limits = [(0, 20), (0, 20)] # The limits of the computational domain.
deltas = [0.4, 0.4] # The dimensions of the grid cells.
hdens_contour = HighestDensityContour(mul_dist, 25, 6, limits, deltas)

# Plot the two contours.
plt.scatter(iform_contour.coordinates[1], iform_contour.coordinates[0],
            label='IFORM contour')
plt.scatter(hdens_contour.coordinates[1], hdens_contour.coordinates[0],
            label='Highest density contour')
plt.xlabel('Zero-up-crossing period, Tz (s)')
plt.ylabel('Significant wave height, Hs (m)')
plt.legend()
plt.show()

# Alternatively, we can plot using viroconcom's plot_contour() function.
plot_contour(iform_contour.coordinates[1], iform_contour.coordinates[0],
             x_label='Tz (s)', y_label='Hs (m)')
plt.show()
