import numpy as np
import matplotlib.pyplot as plt

from viroconcom.distributions import ExponentiatedWeibullDistribution
from viroconcom.read_write import read_ecbenchmark_dataset

# Load sea state measurements from the NDBC buoy 44007.
hs, tz, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

# Fit the exponentiated Weibull distribution to the measurements.
dist = ExponentiatedWeibullDistribution()
params = dist.fit(hs)

# Print the distribution object, which outputs the estimated parameters.
print(dist)

# Plot the density of the empirical distribution and the fitted distribution.
fig, ax = plt.subplots(1, 1)
plt.hist(hs, bins=20, density=True, label='Empirical distribution')
x = np.linspace(0, max(hs), 200)
plt.plot(x, dist.pdf(x), label='Fitted distribution')
plt.xlabel(label_hs)
plt.ylabel('Probability density (-)')
ax.legend()
plt.show()
