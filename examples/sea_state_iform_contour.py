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
# The syntax used to define model structures is described in the documentation.
dist_description_hs = {'name': 'Weibull_Exp'}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0),
                      'functions': ('asymdecrease3', None, 'lnsquare2')}
model_structure = (dist_description_hs, dist_description_tz)

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), model_structure)
fitted_distribution = fit.mul_var_dist


# Compute an IFORM contour with a return period of 50 years.
tr = 50 # Return period in years.
ts = 1 # Sea state duration in hours.
contour = IFormContour(fitted_distribution, tr, ts)

# Plot the data and the contour.
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(sample_tz, sample_hs, c='black', alpha=0.5)
plot_contour(contour.coordinates[1], contour.coordinates[0],
             ax=ax, x_label=label_tz, y_label=label_hs)
plt.show()
