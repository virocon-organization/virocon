import numpy as np

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
prng = np.random.RandomState(42)
sample_1 = prng.normal(10, 1, 500)
sample_2 = [point + prng.uniform(-5, 5) for point in sample_1]
dist_description_1 = {'name': 'KernelDensity', 'dependency': (None, None, None), 'number_of_intervals': 5}
dist_description_2 = {'name': 'Normal', 'dependency': (None, 0, None), 'functions':(None, 'f1', None)}
my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2), timeout=None)
#my_contour = IFormContour(my_fit.mul_var_dist)
#example_plot = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IForm")
