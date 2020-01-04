import csv
import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour


def read_dataset(path='examples/datasets/A.txt'):
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

sample_hs, sample_tz, label_hs, label_tz = read_dataset()

# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38 and that is called 'conditonal modeling approach' (CMA).
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('exp3', None, 'power3') #Shape, Location, Scale
                      }

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), (
    dist_description_hs, dist_description_tz))
dist0 = fit.mul_var_dist.distributions[0]
print('First variable: ' + dist0.name + ' with '
      + ' scale: ' + str(dist0.scale) + ', '
      + ' shape: ' + str(dist0.shape) + ', '
      + ' location: ' + str(dist0.loc))
print('Second variable: ' + str(fit.mul_var_dist.distributions[1]))
