from virocon import ScipyDistribution

import scipy.stats as sts

import matplotlib.pyplot as plt
import numpy as np


# Generalized Gamma Distribution
scipy_dist_name = "gengamma"
scipy_dist = sts.gengamma

sample_params = {"a": 1.5, "c": 2, "loc": 0, "scale": 0.5}
n = 1000
sample = scipy_dist.rvs(**sample_params, size=n, random_state=42)
plt.close("all")
x = np.linspace(sample.min(), sample.max())
sample.sort()
events = sample[:, np.newaxis]
ecdf = (sample <= events).sum(axis=-1) / n
p = np.linspace(0, 1, num=n)
q = np.quantile(sample, p)


class GeneralizedGammaDistributionByName(ScipyDistribution):

    scipy_dist_name = scipy_dist_name


class GeneralizedGammaDistributionByDist(ScipyDistribution):

    scipy_dist = scipy_dist


# For the 3 parameter variant fix the loc parameter when instantiating

try:
    gengamma_by_name = GeneralizedGammaDistributionByName(floc=0)
except TypeError:
    pass
try:
    gengamma_by_dist = GeneralizedGammaDistributionByDist(floc=0)
except TypeError:
    pass

gengamma_by_name = GeneralizedGammaDistributionByName(f_loc=0)
gengamma_by_dist = GeneralizedGammaDistributionByDist(f_loc=0)


# Done

# Fit
gengamma_by_name.fit(sample)
gengamma_by_dist.fit(sample)

assert list(gengamma_by_name.parameters.keys()) == ["a", "c", "loc", "scale"]
assert list(gengamma_by_dist.parameters.keys()) == ["a", "c", "loc", "scale"]

np.testing.assert_allclose(
    gengamma_by_name.parameters["a"], sample_params["a"], rtol=0.1
)
np.testing.assert_allclose(
    gengamma_by_name.parameters["c"], sample_params["c"], rtol=0.1
)
assert gengamma_by_name.parameters["loc"] == 0.0
np.testing.assert_allclose(
    gengamma_by_name.parameters["scale"], sample_params["scale"], rtol=0.1
)

np.testing.assert_allclose(
    gengamma_by_name.parameters["a"], sample_params["a"], rtol=0.1
)
np.testing.assert_allclose(
    gengamma_by_name.parameters["c"], sample_params["c"], rtol=0.1
)
assert gengamma_by_name.parameters["loc"] == 0.0
np.testing.assert_allclose(
    gengamma_by_name.parameters["scale"], sample_params["scale"], rtol=0.1
)


# sample
sample_mean = sample.mean()
sample_var = sample.var()

sample_by_name = gengamma_by_name.draw_sample(n, random_state=42)
sample_by_dist = gengamma_by_dist.draw_sample(n, random_state=42)
mean_by_name = sample_by_name.mean()
var_by_name = sample_by_name.var()
mean_by_dist = sample_by_dist.mean()
var_by_dist = sample_by_dist.var()
assert mean_by_name == mean_by_dist
assert var_by_name == var_by_dist

np.testing.assert_allclose(sample_mean, mean_by_name, rtol=0.1)
np.testing.assert_allclose(sample_var, var_by_name, rtol=0.1)

# pdf
pdf_by_name = gengamma_by_name.pdf(x)
pdf_by_dist = gengamma_by_dist.pdf(x)
fig, ax = plt.subplots()
ax.hist(
    sample,
    bins="doane",
    density=True,
    color="k",
    alpha=0.2,
    histtype="stepfilled",
    label="histogram",
)
ax.plot(x, pdf_by_name, label="by_name")
ax.plot(x, pdf_by_dist, label="by_dist")
ax.legend()
assert (pdf_by_name == pdf_by_dist).all()
assert (scipy_dist.pdf(x, **gengamma_by_name.parameters) == pdf_by_name).all()

# cdf
cdf_by_name = gengamma_by_name.cdf(x)
cdf_by_dist = gengamma_by_dist.cdf(x)
fig, ax = plt.subplots()
ax.step(sample, ecdf, color="k", alpha=0.5, label="ecdf")
ax.plot(x, cdf_by_name, label="by_name")
ax.plot(x, cdf_by_dist, label="by_dist")
ax.legend()
assert (cdf_by_name == cdf_by_dist).all()
assert (scipy_dist.cdf(x, **gengamma_by_name.parameters) == cdf_by_name).all()

# icdf
icdf_by_name = gengamma_by_name.icdf(p)
icdf_by_dist = gengamma_by_dist.icdf(p)
fig, ax = plt.subplots()
ax.step(p, q, color="k", alpha=0.5, label="quantiles")
ax.plot(p, icdf_by_name, label="by_name")
ax.plot(p, icdf_by_dist, label="by_dist")
ax.legend()
assert (icdf_by_name == icdf_by_dist).all()
assert (scipy_dist.ppf(p, **gengamma_by_name.parameters) == icdf_by_name).all()
