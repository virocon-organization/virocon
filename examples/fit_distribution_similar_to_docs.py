import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour


prng = np.random.RandomState(42)

# Draw 1000 samples from a Weibull distribution with shape=1.5 and scale=3,
# which represents significant wave height.
sample_1 = prng.weibull(1.5, 1000)*3

# Let the second sample, which represents spectral peak period increase
# with significant wave height and follow a Lognormal distribution with
# mean=2 and sigma=0.2
sample_2 = [0.1 + 1.5 * np.exp(0.2 * point) +
            prng.lognormal(2, 0.2) for point in sample_1]

plt.scatter(sample_1, sample_2)
plt.xlabel('Significant wave height (m)')
plt.ylabel('Spectral peak period (s)')
plt.show()

# Describe the distribution that should be fitted to the sample.
dist_description_0 = {'name': 'Weibull',
                      'dependency': (None, None, None),
                      'width_of_intervals': 2}
dist_description_1 = {'name': 'Lognormal',
                      'dependency': (None, None, 0),
                      'functions': (None, None, 'exp3')}

# Compute the fit.
my_fit = Fit((sample_1, sample_2),
             (dist_description_0, dist_description_1))

# Plot the fit for the significant wave height, Hs.
# For panel A: use a histogram.
fig = plt.figure(figsize=(9, 4.5))
ax_1 = fig.add_subplot(121)
param_grid = my_fit.multiple_fit_inspection_data[0].scale_at
plt.hist(my_fit.multiple_fit_inspection_data[0].scale_samples[0], density=1,
         label='sample')
shape = my_fit.mul_var_dist.distributions[0].shape(0)
scale = my_fit.mul_var_dist.distributions[0].scale(0)
plt.plot(np.linspace(0, 20, 100),
         sts.weibull_min.pdf(np.linspace(0, 20, 100), c=shape, loc=0,
                             scale=scale),
         label='fitted Weibull distribution')
plt.xlabel('Significant wave height (m)')
plt.ylabel('Probability density (-)')
plt.legend()
# For panel B: use a Q-Q plot.
ax_2 = fig.add_subplot(122)
sts.probplot(my_fit.multiple_fit_inspection_data[0].scale_samples[0],
             sparams=(shape, 0, scale), dist=sts.weibull_min, plot=plt)
ax_2.get_lines()[0].set_markerfacecolor('#1f77ba') # Adapt to v2.0 colors
ax_2.get_lines()[0].set_markeredgecolor('#1f77ba') # Adapt to v2.0 colors
ax_2.get_lines()[1].set_color('#ff7f02') # Adapt to v2.0 colors
plt.title("")
plt.xlabel('Theoretical quantiles (m)')
plt.ylabel('Data quantiles (m)')
plt.show()

# Plot the fits for the spectreal peak period, Tp.
fig = plt.figure(figsize=(10, 8))
ax_1 = fig.add_subplot(221)
title1 = ax_1.set_title('Tp-Distribution for 0≤Hs<2')
param_grid = my_fit.multiple_fit_inspection_data[1].scale_at
ax1_hist = ax_1.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[0], density=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[0])
ax1_plot = ax_1.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))

ax_2 = fig.add_subplot(222)
title2 = ax_2.set_title('Tp-Distribution for 2≤Hs<4')
ax2_hist = ax_2.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[1], density=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[1])
ax2_plot = ax_2.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))

ax_3 = fig.add_subplot(223)
title3 = ax_3.set_title('Tp-Distribution for 4≤Hs<6')
ax3_hist = ax_3.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[2], density=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[2])
ax3_plot = ax_3.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax_3.set_xlabel('spectral peak period [s]')

ax_4 = fig.add_subplot(224)
title4 = ax_4.set_title('Tp-Distribution for 6≤Hs<8')
ax4_hist = ax_4.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[3], density=1)
shape = my_fit.mul_var_dist.distributions[1].shape(0)
scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[3])
ax4_plot = ax_4.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax_4.set_xlabel('Spectral peak period (s)')
plt.show()

# Plot the fit of the dependency function of scale.
fig = plt.figure()
x_1 = np.linspace(0, 12, 100)
plt.plot(param_grid, my_fit.multiple_fit_inspection_data[1].scale_value, 'x',
         label='Discrete scale values')
plt.plot(x_1, my_fit.mul_var_dist.distributions[1].scale(x_1),
         label='Fitted dependency function')
plt.xlabel('Significant wave height (m)')
plt.ylabel('Scale parameter of the Tp distribution (-)')
plt.legend()
plt.show()


# Compute a contour based on the fit and plot it together with the sample.
iform_contour = IFormContour(my_fit.mul_var_dist, 25, 3, 100)
plt.scatter(sample_1, sample_2, label='sample')
plt.plot(iform_contour.coordinates[0], iform_contour.coordinates[1],
            '-k', label='IFORM contour')
plt.xlabel('Significant wave height (m)')
plt.ylabel('Spectral peak period (s)')
plt.legend()
plt.show()
