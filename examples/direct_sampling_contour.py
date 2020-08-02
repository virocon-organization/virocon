from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import DirectSamplingContour
import matplotlib.pyplot as plt

# Define a Weibull distribution representing significant wave height.
dist0 = WeibullDistribution(shape=1.471, loc=0.8888, scale=2.776)
dep0 = (None, None, None)  # All three parameters are independent.

# Define a lognormal distribution representing spectral peak period.
my_sigma = FunctionParam("exp3", 0.0400, 0.1748, -0.2243)
my_mu = FunctionParam("power3", 0.1, 1.489, 0.1901)
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0)  # Parameter one and three depend on dist0.

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute a 1-year direct sampling contour based on drawing 10^6 observations.
contour = DirectSamplingContour(mul_var_dist=mul_dist, return_period=1,
                                state_duration=6, n=1000000, deg_step=6)

# Plot the contour and the sample.
plt.scatter(contour.sample[1], contour.sample[0], marker='.')
plt.plot(contour.coordinates[1], contour.coordinates[0], color='red')
plt.plot([contour.coordinates[1][-1], contour.coordinates[1][0]],
         [contour.coordinates[0][-1], contour.coordinates[0][0]], color='red')
plt.title('1-year direct sampling contour')
plt.ylabel('Significant wave height (m)')
plt.xlabel('Zero-up-crossing period (s)')
plt.show()
