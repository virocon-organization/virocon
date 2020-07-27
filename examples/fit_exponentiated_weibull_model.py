import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.plot import plot_contour


sample_hs, sample_tz, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

# Define the structure of the probabilistic model that will be fitted to the
# dataset. This model structure has been proposed in the paper "Global
# hierarchical models for wind and wave contours: Physical interpretations
# of the dependence functions" by Haselsteiner et al. (2020).
dist_description_hs = {'name': 'Weibull_Exp',
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('asymdecrease3', None, 'lnsquare2'),
                      }

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), (
    dist_description_hs, dist_description_tz))
mul_dist = fit.mul_var_dist
dist0 = fit.mul_var_dist.distributions[0]
print('First variable: ' + dist0.name + ' with '
      + ' scale: ' + str(dist0.scale) + ', '
      + ' shape: ' + str(dist0.shape) + ', '
      + ' location: ' + str(dist0.loc) + ', '
      + ' shape2: ' + str(dist0.shape2))
print('Second variable: ' + str(fit.mul_var_dist.distributions[1]))


# Compute an IFORM contour with a return period of 50 years and a sea
# state duration of 1 hour.
tr = 50 # Return period in years.
ts = 1 # Sea state duration in hours.
contour = IFormContour(mul_dist, tr, ts, 200)

# Plot the data and the contour.
plt.scatter(sample_tz, sample_hs, c='black')
plt.plot(contour.coordinates[1], contour.coordinates[0])
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel(label_tz)
plt.ylabel(label_hs)
plt.show()

#plot_contour(contour.coordinates[1], contour.coordinates[0])
