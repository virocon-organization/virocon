from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, ISormContour, HighestDensityContour
import matplotlib.pyplot as plt

# Define the multivariate distribution given in the paper by Vanem and
# Bitner-Gregersen (2012; doi: 10.1016/j.apor.2012.05.006)
shape = ConstantParam(1.471)
loc = ConstantParam(0.889)
scale = ConstantParam(2.776)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.
my_sigma = FunctionParam(0.040, 0.175, -0.224, "exp3")
my_mu = FunctionParam(0.100, 1.489, 0.190, "power3")
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute an IFORM, an ISORM and a highest density contour.
return_period = 50 # In years
sea_state_duration = 6 # In hours
iform_contour = IFormContour(mul_dist, return_period, sea_state_duration, 100)
isorm_contour = ISormContour(mul_dist, return_period, sea_state_duration, 100)
limits = [(0, 20), (0, 20)] # Limits of the computational domain
deltas = [0.005, 0.005] # Dimensions of the grid cells
hdens_contour = HighestDensityContour(
    mul_dist, return_period, sea_state_duration, limits, deltas)

# Plot the three contours.
plt.scatter(hdens_contour.coordinates[0][0], hdens_contour.coordinates[0][1],
            label="highest density contour")
plt.scatter(iform_contour.coordinates[0][0], iform_contour.coordinates[0][1],
            label="IFORM contour")
plt.scatter(isorm_contour.coordinates[0][0], isorm_contour.coordinates[0][1],
            label="ISORM contour")
plt.xlabel('significant wave height [m]')
plt.ylabel('zero-upcrossing period [s]')
plt.legend()
plt.show()


# To evalute viroconcom, we compare the maximum values of the contour with the
# results from Haselsteiner et al. (2017; doi: 10.1016/j.coastaleng.2017.03.002).
#
# Results in Haselsteiner et al. (2017) with alpha = 1.37 * 10^-5 are:
# Method           maximum Hs (m)     maximum Tz (s)
# IFORM contour    15.23              13.96
# ISORM contour    ca. 17.4           ca. 14.9
# HDC              16.79              14.64
print('Maximum values for the IFORM contour: ' +
      '{:.2f}'.format(max(iform_contour.coordinates[0][0])) + ' m, '
      + '{:.2f}'.format(max(iform_contour.coordinates[0][1])) + ' s')
print('Maximum values for the ISORM contour: ' +
      '{:.2f}'.format(max(isorm_contour.coordinates[0][0])) + ' m, '
      + '{:.2f}'.format(max(isorm_contour.coordinates[0][1])) + ' s')
print('Maximum values for the HDC: ' +
      '{:.2f}'.format(max(hdens_contour.coordinates[0][0])) + ' m, '
      + '{:.2f}'.format(max(hdens_contour.coordinates[0][1])) + ' s')
