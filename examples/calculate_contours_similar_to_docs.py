from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, \
    MultivariateDistribution
from viroconcom.contours import IFormContour, HighestDensityContour
import matplotlib.pyplot as plt

# Define a Weibull distribution representing significant wave height.
shape = ConstantParam(1.5)
loc = ConstantParam(1)
scale = ConstantParam(3)
dist0 = WeibullDistribution(shape, loc, scale)
dep0 = (None, None, None) # All three parameters are independent.

# Define a Lognormal distribution representing spectral peak period.
my_sigma = FunctionParam(0.05, 0.2, -0.2, "exp3")
my_mu = FunctionParam(0.1, 1.5, 0.2, "power3")
dist1 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep1 = (0, None, 0) # Parameter one and three depend on dist0.

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1]
dependencies = [dep0, dep1]
mul_dist = MultivariateDistribution(distributions, dependencies)

#print(mul_dist.draw_multivariate_sample(100))
import numpy as np
x, y = mul_dist.draw_multivariate_sample(100)

x, y = mul_dist.direct_sampling_contour(x, y, 0.1, 5)
print(x,y)
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# use the previously defined function
scatter_hist(x, y, ax, ax_histx, ax_histy)

plt.show()


# Compute an IFORM contour with a return period of 25 years, a sea state
# duration of 3 hours and 100 points along the contour.
iform_contour = IFormContour(mul_dist, 25, 3, 100)

# Compute a highest density contour with the same settings (25 years return
# period, 3 hour sea state duration).
limits = [(0, 20), (0, 20)] # The limits of the computational domain.
deltas = [0.4, 0.4] # The dimensions of the grid cells.
hdens_contour = HighestDensityContour(mul_dist, 25, 3, limits, deltas)

# Plot the two contours.
plt.scatter(hdens_contour.coordinates[0][0], hdens_contour.coordinates[0][1],
            label="highest density contour")
plt.scatter(iform_contour.coordinates[0][0], iform_contour.coordinates[0][1],
            label="IFORM contour")
plt.xlabel('significant wave height [m]')
plt.ylabel('spectral peak period [s]')
plt.legend()
plt.show()
