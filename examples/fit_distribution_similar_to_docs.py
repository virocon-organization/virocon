import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
prng = np.random.RandomState(42)
sample_1 = prng.normal(10, 1, 500)
sample_2 = [point + prng.uniform(-5, 5) for point in sample_1]
#dist_description_1 = {'name': 'KernelDensity', 'dependency': (None, None, None), 'number_of_intervals': 5}
#dist_description_2 = {'name': 'Normal', 'dependency': (None, 0, None), 'functions':(None, 'f1', None)}
#my_fit = Fit((sample_1, sample_2), (dist_description_1, dist_description_2), timeout=None)
#my_contour = IFormContour(my_fit.mul_var_dist, timeout=None)
#example_plot = plt.scatter(my_contour.coordinates[0][0], my_contour.coordinates[0][1], label="IFORM contour")
#plt.show()


dist_description_0 = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 3}
dist_description_1 = {'name': 'Lognormal_1', 'dependency': (None, None, 0), 'functions': (None, None, 'f2')}
my_fit = Fit((sample_1, sample_2), (dist_description_0, dist_description_1), timeout=None)
print(my_fit)
print(my_fit.multiple_fit_inspection_data[1].scale_at)
print(my_fit.multiple_fit_inspection_data[1].scale_value)
print(my_fit.multiple_fit_inspection_data[1].shape_at)
print(my_fit.multiple_fit_inspection_data[1].shape_value)
print(my_fit.multiple_fit_inspection_data[1].loc_at)
print(my_fit.multiple_fit_inspection_data[1].loc_value)

