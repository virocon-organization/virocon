import numpy as np


def direct_sampling_contour(x, y, w, probability, d_s_deg):
    """
    calculates direct sampling contour
    for fast compution, the data should be 100000 points or less
    Parameters
    ----------
    x,y : array like
        sample of data
    probability : float
        non exceedance probability of contour
    d_s_deg : float
        directional step in degrees

    Returns
    -------
    x_con, y_con :
        contour of sample
    """
    #print(w)
    dt = d_s_deg * np.pi / 180
    dv = d_s_deg * np.pi / 180
    transposed1 = np.transpose(np.arange(dt, 2 * np.pi, dt))
    transposed2 = np.transpose(np.arange(dv, 2 * np.pi, dv))
    length_x = len(x)
    length_t = len(transposed1)
    r = np.zeros(length_t)

    # find radius for each angle
    i = 0
    j = 0
    if length_x >= 1000001:
        raise RuntimeWarning('Takes longer then normal. Maybe use fewer data.')
    else:
        while i < length_t:

            z = x * np.cos(transposed1[i]) * np.cos(transposed2[i]) + y * np.sin(transposed1[i]) * np.cos(transposed2[i]) + w * np.sin(transposed2[i])
            r[i] = np.quantile(z, probability)
            i = i + 1

    #print(z)
    #print(r)
    # find intersection of lines
    t1 = np.array(np.concatenate((transposed1, [dt]), axis=0))
    t2 = np.array(np.concatenate((transposed2, [dv]), axis=0))
    r = np.array(np.concatenate((r, [r[0]]), axis=0))

    denominator = (np.sin(t1[2:]) * np.cos(t1[1:len(t1) - 1]) - np.sin(t1[1:len(t1) - 1])* np.cos(t1[2:])) * np.cos(t2[1:len(t2) - 1])*(np.cos(t2[1:len(t2) - 1])* np.sin(t2[2:]) - np.sin(t2[1:len(t2) - 1])* np.cos(t2[2:]))

    c1 = r[2:]*np.cos(t1[1:len(t1) - 1]) * np.cos(t2[1:len(t2) - 1]) + r[2:]*np.sin(t1[1:len(t1) - 1])*np.cos(t2[1:len(t2) - 1]) + r[2:]*np.sin(t2[1:len(t2) - 1])
    c2 = r[1:len(r)-1]*np.cos(t1[2:]) * np.cos(t2[1:len(t2) - 1]) + r[1:len(r)-1]*np.sin(t1[2:])*np.cos(t2[1:len(t2) - 1]) + r[2:]*np.sin(t2[1:len(t2) - 1])
    c3 = r[2:]*np.cos(t1[1:len(t1) - 1]) * np.cos(t2[2:]) + r[2:]*np.sin(t1[1:len(t1) - 1])*np.cos(t2[2:]) + r[1:len(r)-1]*np.sin(t2[2:])

    ax = np.sin(t1[2:])*np.cos(t2[1:len(t2)-1])*np.sin(t2[2:]) - np.sin(t1[1:len(t1)-1])*np.sin(t2[1:len(t2)-1])*np.cos(t2[2:])
    bx = np.sin(t1[1:len(t1)-1])*(np.sin(t2[1:len(t2)-1])*np.cos(t2[2:])-np.cos(t2[1:len(t2)-1])*np.sin(t2[2:]))
    cx = -(np.sin(t1[2:])-np.sin(t1[1:len(t1)-1]))*np.cos(t2[1:len(t2)-1])*np.sin(t2[1:len(t2)-1])

    ay = np.cos(t1[1:len(t1)-1])*np.sin(t2[1:len(t2)-1])*np.cos(t2[2:])-np.cos(t1[2:])*np.cos(t2[1:len(t2)-1])*np.sin(t2[2:])
    by = np.cos(t1[1:len(t1)-1])*(np.cos(t2[1:len(t2)-1])*np.sin(t2[2:])-np.sin(t2[1:len(t2)-1])*np.cos(t2[2:]))
    cy = (np.cos(t1[2:])-np.cos(t1[1:len(t1)-1]))*np.cos(t2[1:len(t2)-1])*np.sin(t2[1:len(t2)-1])

    x_cont = (ax*c1+bx*c2+cx*c3) / denominator
    y_cont = (ay*c1+by*c2+cy*c3) / denominator
    z_cont = (np.cos(t2[2:])*c1-np.cos(t2[1:len(t2)-1])*c3) / (np.sin(t2[1:len(t2)-1])*np.cos(t2[2:])-np.cos(t2[1:len(t2)-1])*np.sin(t2[2:]))

    return x_cont, y_cont, z_cont


from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import WeibullDistribution, LognormalDistribution, MultivariateDistribution

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

my_sigma = FunctionParam(0.07, 0.1, -0.5, "exp3")
my_mu = FunctionParam(0.05, 1, 0.4, "power3")
dist2 = LognormalDistribution(sigma=my_sigma, mu=my_mu)
dep2 = (0, None, 0)

# Create a multivariate distribution by bundling the two distributions.
distributions = [dist0, dist1, dist2]
dependencies = [dep0, dep1, dep2]
mul_dist = MultivariateDistribution(distributions, dependencies)

# Draw sample from multivariate distribution with given number.
n = 100000 # Number of how many data is to be drawn for the sample.
sample = mul_dist.draw_multivariate_sample(n)

# Compute a direct sampling contour
# probability of 1 percent, step of 5 degrees
direct_sampling_contour = direct_sampling_contour(sample[0], sample[1], sample[2], 0.1, 5)

# Plot the contour
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
x,y,z=direct_sampling_contour
ax.contour(x,y,z,50)
plt.show()