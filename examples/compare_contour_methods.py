from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, ISormContour, \
    HighestDensityContour, DirectSamplingContour, sort_points_to_form_continous_line
from viroconcom.plot import plot_contour
import matplotlib.pyplot as plt

# Define the multivariate distribution given in the paper by Vanem and
# Bitner-Gregersen (2012; doi: 10.1016/j.apor.2012.05.006)
shape = ConstantParam(1.471)
loc = ConstantParam(0.889)
scale = ConstantParam(2.776)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.
my_sigma = FunctionParam('exp3', 0.040, 0.175, -0.224)
my_mu = FunctionParam('power3', 0.100, 1.489, 0.190)
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Compute an IFORM, an ISORM, a direct sampling and a highest density contour.
return_period = 50 # In years
sea_state_duration = 6 # In hours
iform_contour = IFormContour(mul_dist, return_period, sea_state_duration, 100)
isorm_contour = ISormContour(mul_dist, return_period, sea_state_duration, 100)
ds_contour = DirectSamplingContour(
    mul_dist, return_period, sea_state_duration, 5000000)
limits = [(0, 20), (0, 20)] # Limits of the computational domain
deltas = [0.005, 0.005] # Dimensions of the grid cells
hdens_contour = HighestDensityContour(
    mul_dist, return_period, sea_state_duration, limits, deltas)
hdc_coordinates = sort_points_to_form_continous_line(
    hdens_contour.coordinates[0][0], hdens_contour.coordinates[0][1])

# Plot the four contours (a similar plot was presented in the paper by
# Haselsteiner et al. (2017; 10.1016/j.coastaleng.2017.03.002), Fig. 8 c).
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
plot_contour(iform_contour.coordinates[0][1], iform_contour.coordinates[0][0],
             ax, line_style='b-', contour_label='IFORM contour')
plot_contour(isorm_contour.coordinates[0][1], isorm_contour.coordinates[0][0],
             ax, line_style='r-', contour_label='ISORM contour')
plot_contour(ds_contour.coordinates[1], ds_contour.coordinates[0], ax,
             line_style='g-', contour_label='Direct sampling contour')
plot_contour(hdc_coordinates[1], hdc_coordinates[0], ax,
             line_style='k-', contour_label='Highest density contour')
plt.xlabel('Zero-up-crossing period (s)')
plt.ylabel('Significant wave height (m)')
plt.legend()
plt.show()
