import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.plot import plot_dependence_functions

# Load wind-wave data from the hindcast coastDat-2.
v, hs, label_v, label_hs = read_ecbenchmark_dataset('datasets/1year_dataset_D.txt')

# Define the model structure for wind-wave states that was proposed at OMAE2020.
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

# Plot the dependence functions and density lines of the fitted distribution.
fig2, axs = plt.subplots(1, 3, figsize=(13, 4.5))
plot_dependence_functions(fit, fig2, ax1=axs[0], ax2=axs[1],
                          unconditonal_variable_label=label_v,
                          factor_draw_longer=1.6)
x = np.linspace(0, 35, 100)
y = np.linspace(0, 18, 100)
X, Y = np.meshgrid(x, y)
f = np.ndarray(shape=X.shape, dtype=float)
for i, xi in enumerate(X):
    f[i] = fit.mul_var_dist.pdf_2d(xi, Y[i])
axs[2].scatter(v, hs, c='black', alpha=0.5)
axs[2].contour(X, Y, f, [0.000001, 0.00001, 0.0001, 0.001, 0.01], colors='red')
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].set_xlabel(label_v)
axs[2].set_ylabel(label_hs)
plt.show()
