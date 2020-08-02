import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.plot import SamplePlotData, plot_dependence_functions, plot_sample

# Load sea state measurements from the NDBC buoy 44007.
hs, tz, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

# Describe the distribution that should be fitted to the sample.
# This model structure for sea states was proposed in the OMAE2020 paper.
dist_description_hs = {'name': 'Weibull_Exp',
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0),
                      'functions': ('asymdecrease3', None, 'lnsquare2'),
                      'min_datapoints_for_fit': 50}

# Fit the model structure to the buoy data.
fit = Fit((hs, tz), (dist_description_hs, dist_description_tz))

# Plot the fitted dependence functions.
fig1, axs = plt.subplots(1, 3, figsize=(13, 4.5))
plot_dependence_functions(fit, fig1, ax1=axs[0], ax2=axs[1],
                          unconditonal_variable_label=label_hs,
                          factor_draw_longer=2)

# Plot lines of lines of constant density of the fitted model.
x = np.linspace(0, 15, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)
f = np.ndarray(shape=X.shape, dtype=float)
for i, xi in enumerate(X):
    f[i] = fit.mul_var_dist.pdf_2d(xi, Y[i])
plt.subplot(133)
plt.scatter(tz, hs, c='black', alpha=0.5)
plt.contour(Y, X, f, [0.000001, 0.00001, 0.0001, 0.001, 0.01], colors='red')
plt.xlabel(label_tz)
plt.ylabel(label_hs)
plt.show()

# Load wind-wave data from the hindcast coastDat-2.
v, hs, label_v, label_hs = \
    read_ecbenchmark_dataset('datasets/1year_dataset_D.txt')

# Describe the distribution that should be fitted to the sample.
# This model structure for wind-wave states was proposed in the OMAE2020 paper.
dist_description_v =  {'name': 'Weibull_Exp',
                       'width_of_intervals': 2}
dist_description_hs = {'name': 'Weibull_Exp',
                       'fixed_parameters': (None, None, None, 5),
                       'dependency': (0, None, 0, None),
                       'functions': ('logistics4', None, 'alpha3', None),
                       'min_datapoints_for_fit': 50,
                       'do_use_weights_for_dependence_function': True}

# Fit the model structure to the hindcast data.
fit = Fit((v, hs), (dist_description_v, dist_description_hs))

# Plot the fitted dependence functions.
fig2, axs = plt.subplots(1, 3, figsize=(13, 4.5))
plot_dependence_functions(fit, fig2, ax1=axs[0], ax2=axs[1],
                          unconditonal_variable_label=v,
                          factor_draw_longer=2)

# Plot lines of lines of constant density of the fitted model.
x = np.linspace(0, 40, 100)
y = np.linspace(0, 15, 100)
X, Y = np.meshgrid(x, y)
f = np.ndarray(shape=X.shape, dtype=float)
for i, xi in enumerate(X):
    f[i] = fit.mul_var_dist.pdf_2d(np.array(xi), np.array(Y[i]))
plt.subplot(133)
plt.scatter(v, hs, c='black', alpha=0.5)
plt.contour(X, Y, f, [0.000001, 0.00001, 0.0001, 0.001, 0.01], colors='red')
plt.xlabel(label_v)
plt.ylabel(label_hs)
plt.show()

