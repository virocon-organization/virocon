from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
import numpy as np

prng = np.random.RandomState(42)

# Draw 1000 observations from a Weibull distribution with
# shape=1.5 and scale=3, which represents significant
# wave height.
sample_0 = prng.weibull(1.5, 1000) * 3

# Let the second sample, which represents spectral peak
# period, increase with significant wave height and follow
# a lognormal distribution with sigma=0.2.
sample_1 = [0.1 + 1.5 * np.exp(0.2 * point) +
            prng.lognormal(2, 0.2) for point in sample_0]

# Define a bivariate probabilistic model that will be fitted to
# the samples. Set a parametric distribution for each variable and
# a dependence structure. Set the lognormal distribution's scale
# parameter to depend on the variable with index 0, which represents
# significant wave height by using the 'dependency' key-value pair.
# A 3-parameter exponential function is chosen to define the
# dependency by setting the function to 'exp3'. The dependency for
# the parameters must be given in the order shape, location, scale.
dist_description_0 = {'name': 'Weibull',
                      'dependency': (None, None, None),
                      'width_of_intervals': 2}
dist_description_1 = {'name': 'Lognormal_ShapeNoneScale',
                      'dependency': (None, None, 0),
                      'functions': (None, None, 'exp3')}

# Compute the fit.
my_fit = Fit((sample_0, sample_1),
             (dist_description_0, dist_description_1))

# Compute an environmental contour with a return period of
# 25 years and a sea state duration of 3 hours. 100 data points
# along the contour shall be calculated.
iform_contour = IFormContour(my_fit.mul_var_dist, 25, 3, 100)
