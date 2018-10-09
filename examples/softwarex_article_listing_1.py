from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
import numpy as np

prng = np.random.RandomState(42)

# Draw 1000 samples from a Weibull distribution with
# shape=1.5 and scale=3, which represents significant
# wave height.
sample_1 = prng.weibull(1.5, 1000) * 3

# Let the second sample, which represents spectral peak
# period, increase with significant wave height and follow
# a Lognormal distribution with sigma=0.2.
sample_2 = [0.1 + 1.5 * np.exp(0.2 * point) +
            prng.lognormal(2, 0.2) for point in sample_1]

# Describe the distribution that should be fitted to the
# sample.
dist_description_0 = {'name': 'Weibull',
                      'dependency': (None, None, None),
                      'width_of_intervals': 2}
dist_description_1 = {'name': 'Lognormal_1',
                      'dependency': (None, None, 0),
                      'functions': (None, None, 'exp3')}

# Compute the fit.
my_fit = Fit((sample_1, sample_2),
             (dist_description_0, dist_description_1))

# Compute an environmental contour with a return period of
# 25 years and a sea state duration of 3 hours.
iform_contour = IFormContour(my_fit.mul_var_dist, 25, 3, 100)
