from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, MultivariateDistribution
from viroconcom.contours import DirectSamplingContour
import matplotlib.pyplot as plt

# Define a Weibull distribution representing significant wave height.
shape = ConstantParam(1.5)
loc = ConstantParam(1)
scale = ConstantParam(3)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.

# Define a Lognormal distribution representing spectral peak period.
my_sigma = FunctionParam(0.05, 0.2, -0.2, "exp3")
my_mu = FunctionParam(0.1, 1.5, 0.2, "power3")
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Draw sample from multivariate distribution with given number.
n = 100000 # Number of how many data is to be drawn for the sample.
sample = mul_dist.draw_multivariate_sample(n)

# Compute a direct sampling contour
# probability of 1 percent, step of 5 degrees
direct_sampling_contour = DirectSamplingContour.direct_sampling_contour(DirectSamplingContour, sample[0], sample[1], 0.1, 5)

# Plot the contour
plt.plot(direct_sampling_contour[0], direct_sampling_contour[1])
plt.xlabel('significant wave height [m]')
plt.ylabel('spectral peak period [s]')
plt.show()