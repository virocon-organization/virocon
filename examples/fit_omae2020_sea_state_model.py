import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.plot import plot_dependence_functions

# Load sea state measurements from the NDBC buoy 44007.
hs, tz, label_hs, label_tz = read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

# Define the model structure for sea states that was proposed at OMAE2020.
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

# Plot lines of constant density of the fitted model.
x = np.linspace(0, 14, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)
f = np.ndarray(shape=X.shape, dtype=float)
for i, xi in enumerate(X):
    f[i] = fit.mul_var_dist.pdf_2d(xi, Y[i])
axs[2].scatter(tz, hs, c='black', alpha=0.5)
axs[2].contour(Y, X, f, [0.000001, 0.00001, 0.0001, 0.001, 0.01], colors='red')
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].set_xlabel(label_tz)
axs[2].set_ylabel(label_hs)
plt.show()
