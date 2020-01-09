import csv
import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import HighestDensityContour


def read_benchmark_dataset(path='examples/datasets/A.txt'):
    """
    Reads a datasets provided for the environmental contour benchmark.
    Parameters
    ----------
    path : string
        Path to dataset including the file name, defaults to 'examples/datasets/A.txt'
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.
    x_label : str
        Label of the environmantal variable 1.
    y_label : str
        Label of the environmental variable 2.
    """

    x = list()
    y = list()
    x_label = None
    y_label = None
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx == 0:
                x_label = row[1][1:] # Ignore first char (is a white space).
                y_label = row[2][1:] # Ignore first char (is a white space).
            if idx > 0: # Ignore the header
                x.append(float(row[1]))
                y.append(float(row[2]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y, x_label, y_label)

sample_hs, sample_tz, label_hs, label_tz = read_benchmark_dataset()

# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None, None), # Shape, Location, Scale, Shape2
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0), # Shape, Location, Scale
                      'functions': ('exp3', None, 'power3') # Shape, Location, Scale
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


# Compute a highest density contour with a return period of 50 years and
# a sea  state duration of 1 hour.
tr = 50 # Return period in years.
ts = 1 # Sea state duration in hours.
limits = [(0, 20), (0, 20)] # Limits of the computational domain.
deltas = [0.01, 0.01] # Dimensions of the grid cells.
hdens_contour = HighestDensityContour(mul_dist, tr, ts, limits, deltas)

# Plot the data and the contour.
plt.scatter(sample_tz, sample_hs, c='black')
plt.scatter(hdens_contour.coordinates[0][1], hdens_contour.coordinates[0][0])
plt.xlabel('zero-up-crossing period [s]')
plt.ylabel('significant wave height (m)')

plt.legend()
plt.show()
