import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from viroconcom.plot import SamplePlotData, plot_sample, plot_marginal_fit, \
    plot_dependence_functions, plot_contour


sample_0, sample_1, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
sample_plot_data = SamplePlotData(sample_1, sample_0)
plot_sample(sample_plot_data, ax)
plt.xlabel('Zero-up-crossing period (s)')
plt.ylabel('Significant wave height (m)')
plt.show()

# Describe the distribution that should be fitted to the sample.
dist_description_0 = {'name': 'Weibull_Exp',
                      'width_of_intervals': 1}
dist_description_1 = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0),
                      'functions': ('asymdecrease3', None, 'lnsquare2'),
                      'min_datapoints_for_fit': 50
                      }

# Compute the fit.
my_fit = Fit((sample_0, sample_1),
             (dist_description_0, dist_description_1))
fitted_hs_dist = my_fit.mul_var_dist.distributions[0]
fitted_tz_dist = my_fit.mul_var_dist.distributions[1]

# Plot the fitted distribution for the significant wave height with a QQ-plot.
fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
plot_marginal_fit(sample_0, fitted_hs_dist, fig, ax, label='$h_s$ (m)', dataset_char='A')
plt.show()

# Plot fitted marginal distributions for the zero-up-crossing peak period
# using a histogram of the sample and a density plot.
n_fits = len(my_fit.multiple_fit_inspection_data[1].scale_at)
fig, axs = plt.subplots(1, n_fits, figsize=(14, 4))
for i in range(n_fits):
        axs[i].set_title('Tz distribution for ' + str(i) + '≤Hs<' + str(i + 1))
        axs[i].hist(my_fit.multiple_fit_inspection_data[1].scale_samples[i], density=1)
        x = np.linspace(0, 12, 200)
        interval_center = my_fit.multiple_fit_inspection_data[1].scale_at[i]
        f = fitted_tz_dist.pdf(x, np.zeros(x.shape) + interval_center, (0, None, 0))
        axs[i].plot(x, f)
plt.show()

# Plot the fitted dependence functions.
fig = plt.figure(figsize=(9, 4.5))
plot_dependence_functions(my_fit, fig, unconditonal_variable_label=label_hs,
                          factor_draw_longer=2)
plt.show()

# Compute a contour based on the fit and plot it together with the sample.
iform_contour = IFormContour(my_fit.mul_var_dist, 50, 1)
fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
plot_contour(iform_contour.coordinates[1], iform_contour.coordinates[0],
             ax=ax, x_label=label_tz, y_label=label_hs,
             sample_plot_data=sample_plot_data, upper_ylim=13)
plt.show()
