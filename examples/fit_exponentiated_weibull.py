import numpy as np
import matplotlib.pyplot as plt

from viroconcom.distributions import ExponentiatedWeibullDistribution
from viroconcom.read_write import read_ecbenchmark_dataset

hs, tz, label_hs, label_tz = \
    read_ecbenchmark_dataset('datasets/1year_dataset_A.txt')

dist = ExponentiatedWeibullDistribution()
params = dist.fit(hs)

fig, ax = plt.subplots(1, 1)
plt.hist(hs, density=True, label='Empirical distribution')
x = np.linspace(0, max(hs), 200)
plt.plot(x, dist.pdf(x), label='Fitted distribution')
plt.xlabel(label_hs)
plt.ylabel('Probability density (-)')
ax.legend()
plt.show()