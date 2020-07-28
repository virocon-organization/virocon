import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from viroconcom.plot import SamplePlotData, plot_marginal_fit, plot_dependence_functions, plot_contour


sample_0, sample_1, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

plt.scatter(sample_1, sample_0)
plt.xlabel('Zero-up-crossing period (s)')
plt.ylabel('Significant wave height (m)')
plt.show()

# Describe the distribution that should be fitted to the sample.
dist_description_0 = {'name': 'Weibull_Exp',
                      'dependency': (None, None, None, None),
                      'width_of_intervals': 1}
dist_description_1 = {'name': 'Lognormal',
                      'dependency': (0, None, 0),
                      'functions': ('exp3', None, 'power3')}

# Compute the fit.
my_fit = Fit((sample_0, sample_1),
             (dist_description_0, dist_description_1))
fitted_hs_dist = my_fit.mul_var_dist.distributions[0]
fitted_tz_dist = my_fit.mul_var_dist.distributions[1]

# Plot the fit for the significant wave height, Hs.
# For panel A: use a histogram.
fig = plt.figure(figsize=(9, 4.5))
ax_1 = fig.add_subplot(121)
param_grid = my_fit.multiple_fit_inspection_data[0].scale_at
plt.hist(my_fit.multiple_fit_inspection_data[0].scale_samples[0], density=1,
         label='Dataset A')
x = np.linspace(0, 10, 200)
plt.plot(x, fitted_hs_dist.pdf(x),
         label='Fitted Weibull distribution')
plt.xlabel('Significant wave height (m)')
plt.ylabel('Probability density (-)')
plt.legend()
# For panel B: use a Q-Q plot.
ax_2 = fig.add_subplot(122)
plot_marginal_fit(sample_0, fitted_hs_dist, fig, ax_2, dataset_char='A')
plt.show()

# Plot the fits for the spectreal peak period, Tp.
fig = plt.figure(figsize=(10, 8))
ax_1 = fig.add_subplot(221)
title1 = ax_1.set_title('Tz distribution for 0≤Hs<1')
param_grid = my_fit.multiple_fit_inspection_data[1].scale_at
ax1_hist = ax_1.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[0], density=1)
#shape = my_fit.mul_var_dist.distributions[1].shape(0)
#scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[0])
#ax1_plot = ax_1.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
x = np.linspace(0, 12, 200)
ax1_plot = ax_1.plot(x, fitted_tz_dist.pdf(x, rv_values=np.zeros(x.shape) + 0.5, dependencies=(0, None, 0)))

ax_2 = fig.add_subplot(222)
title2 = ax_2.set_title('Tz distribution for 1≤Hs<2')
ax2_hist = ax_2.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[1], density=1)
#shape = my_fit.mul_var_dist.distributions[1].shape(0)
#scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[1])
#ax2_plot = ax_2.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax2_plot = ax_2.plot(x, fitted_tz_dist.pdf(x, rv_values=np.zeros(x.shape) + 1.5, dependencies=(0, None, 0)))

ax_3 = fig.add_subplot(223)
title3 = ax_3.set_title('Tz distribution for 2≤Hs<3')
ax3_hist = ax_3.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[2], density=1)
#shape = my_fit.mul_var_dist.distributions[1].shape(0)
#scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[2])
#ax3_plot = ax_3.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax3_plot = ax_3.plot(x, fitted_tz_dist.pdf(x, rv_values=np.zeros(x.shape) + 2.5, dependencies=(0, None, 0)))
ax_3.set_xlabel(label_tz)

ax_4 = fig.add_subplot(224)
title4 = ax_4.set_title('Tz distribution for 3≤Hs<4')
ax4_hist = ax_4.hist(my_fit.multiple_fit_inspection_data[1].scale_samples[3], density=1)
#shape = my_fit.mul_var_dist.distributions[1].shape(0)
#scale = my_fit.mul_var_dist.distributions[1].scale(param_grid[3])
#ax4_plot = ax_4.plot(np.linspace(0, 20, 100), sts.lognorm.pdf(np.linspace(0, 20, 100), s=shape, scale=scale))
ax4_plot = ax_4.plot(x, fitted_tz_dist.pdf(x, rv_values=np.zeros(x.shape) + 3.5, dependencies=(0, None, 0)))
ax_4.set_xlabel(label_tz)
plt.show()


fig = plt.figure()
plot_dependence_functions(my_fit, fig)

# Compute a contour based on the fit and plot it together with the sample.
iform_contour = IFormContour(my_fit.mul_var_dist, 50, 1)
sample_plot_data = SamplePlotData(sample_1, sample_0)
plot_contour(iform_contour.coordinates[1], iform_contour.coordinates[0],
             x_label=label_tz, y_label=label_hs, sample_plot_data=sample_plot_data)

