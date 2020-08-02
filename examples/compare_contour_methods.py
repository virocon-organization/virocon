from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, ISormContour, \
    HighestDensityContour, DirectSamplingContour, sort_points_to_form_continous_line
from viroconcom.plot import plot_contour
import matplotlib.pyplot as plt

# Four different contours with a return period of 25 years will be constructed.
return_period = 25 # In years
state_duration = 6 # In hours

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
iform_contour = IFormContour(mul_dist, return_period, state_duration)
isorm_contour = ISormContour(mul_dist, return_period, state_duration)
ds_contour = DirectSamplingContour(mul_dist, return_period, state_duration, 5000000)
hdens_contour = HighestDensityContour(mul_dist, return_period, state_duration)
hdc_coordinates = sort_points_to_form_continous_line(
    hdens_contour.coordinates[0], hdens_contour.coordinates[1])

# Plot the four contours (a similar plot was presented in the paper by
# Haselsteiner et al. (2017; 10.1016/j.coastaleng.2017.03.002), Fig. 8 c).
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
plot_contour(iform_contour.coordinates[1], iform_contour.coordinates[0],
             ax, line_style='b-', contour_label='IFORM contour')
plot_contour(isorm_contour.coordinates[1], isorm_contour.coordinates[0],
             ax, line_style='r-', contour_label='ISORM contour')
plot_contour(ds_contour.coordinates[1], ds_contour.coordinates[0], ax,
             line_style='g-', contour_label='Direct sampling contour')
plot_contour(hdc_coordinates[1], hdc_coordinates[0], ax,
             line_style='k-', contour_label='Highest density contour')
plt.xlabel('Zero-up-crossing period (s)')
plt.ylabel('Significant wave height (m)')
plt.legend()
plt.show()


# Method           maximum Hs (m)    maximum Tz (s)    Source (author, year)
# IFORM            14.62             ca. 13.5          Vanem & B.-G. (2012)
# ISORM            ca. 16.8          ca. 14.7          Chai and Leira (2018)
# Direct sampling  14.66             13.68             Huseby, V. & N. (2013)
# HDC              16.18             14.37             Matlab code compute-hdc*
# * https://github.com/ahaselsteiner/compute-hdc
print('Maximum values for the IFORM contour: ' +
      '{:.2f}'.format(max(iform_contour.coordinates[0])) + ' m, '
      + '{:.2f}'.format(max(iform_contour.coordinates[1])) + ' s')
print('Maximum values for the ISORM contour: ' +
      '{:.2f}'.format(max(isorm_contour.coordinates[0])) + ' m, '
      + '{:.2f}'.format(max(isorm_contour.coordinates[1])) + ' s')
print('Maximum values for the direct sampling contour: ' +
      '{:.2f}'.format(max(ds_contour.coordinates[0])) + ' m, '
      + '{:.2f}'.format(max(ds_contour.coordinates[1])) + ' s')
print('Maximum values for the highest density contour: ' +
      '{:.2f}'.format(max(hdens_contour.coordinates[0])) + ' m, '
      + '{:.2f}'.format(max(hdens_contour.coordinates[1])) + ' s')