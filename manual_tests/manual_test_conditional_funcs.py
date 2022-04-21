import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from virocon import (
    get_DNVGL_Hs_Tz,
    get_OMAE2020_Hs_Tz,
    IFORMContour,
    GlobalHierarchicalModel,
    read_ec_benchmark_dataset,
)

import tol_colors

# data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]
data = read_ec_benchmark_dataset("datasets/ec-benchmark_dataset_A.txt")

# dist_description, _, semantics = get_DNVGL_Hs_Tz()
dist_descriptions, fit_descriptions, semantics = get_OMAE2020_Hs_Tz()

model = GlobalHierarchicalModel(dist_descriptions)

model.fit(data)

x = np.linspace(0.5, 15, num=50)
p = np.linspace(0, 1, num=50)
given = 4
dim = 1
given = np.c_[np.full_like(x, given), np.zeros_like(x)]


old_ccdf = model.conditional_cdf(x, dim, given)
new_ccdf = super(type(model), model).conditional_cdf(x, dim, given, random_state=42)

old_cicdf = model.conditional_icdf(p, dim, given)
new_cicdf = super(type(model), model).conditional_icdf(p, dim, given, random_state=42)
# %%

from my_jointmodels import TransformedModel


def identity(x):
    return x


def jac(x):
    return np.ones((len(x)))


t_model = TransformedModel(model, identity, identity, jac)

from my_contours import MyIFORMContour
from virocon import plot_2D_contour

tr = 20  # Return period in years.
ts = 1  # Sea state duration in hours.
alpha = 1 / (tr * 365.25 * 24 / ts)
my_contour = MyIFORMContour(t_model, alpha)
contour = MyIFORMContour(model, alpha)

my_coordinates = my_contour.coordinates
# %%
cset = tol_colors.tol_cset("high-contrast")
plt.close("all")

ax = plot_2D_contour(contour, data, semantics=semantics, swap_axis=True)
# ax = plot_2D_contour(my_contour, data, semantics=semantics, swap_axis=True)
ax.plot(my_coordinates[:, 1], my_coordinates[:, 0], color=cset.yellow)
ax.legend(["data", "ghm", "monte carlo"])


fig, ax = plt.subplots()
ax.plot(x, old_ccdf, label="ghm direct", color=cset[0])
ax.plot(x, new_ccdf, label="monte carlo sample", color=cset[1])
ax.legend()

diff_ccdf = old_ccdf - new_ccdf

fig, ax = plt.subplots()
ax.plot(x, diff_ccdf, label="ghm direct - monte carlo sample", color=cset[0])
ax.axhline(0, color=cset.red)
ax.set_xlabel("Hs")
ax.legend()


old_sf = 1.0 - old_ccdf
new_sf = 1.0 - new_ccdf
rel_sf_diff = (old_sf - new_sf) / old_sf

fig, ax = plt.subplots()
ax.plot(x, old_sf, label="ghm direct", color=cset[0])
ax.plot(x, new_sf, label="monte carlo sample", color=cset[1])
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, rel_sf_diff, label="rel sf diff", color=cset[0])
ax.axhline(0, color=cset.red)
ax.set_xlabel("Hs")
ax.legend()


fig, ax = plt.subplots()
ax.plot(p, old_cicdf, label="ghm direct", color=cset[0])
ax.plot(p, new_cicdf, label="monte carlo sample", color=cset[1])
ax.legend()

diff_cicdf = old_cicdf - new_cicdf

fig, ax = plt.subplots()
ax.plot(p, diff_cicdf, label="ghm direct - monte carlo sample", color=cset[0])
ax.axhline(0, color=cset.red)
ax.set_ylabel("Hs")
ax.legend()

plt.show()

# np.testing.assert_allclose(old_ccdf, new_ccdf)
