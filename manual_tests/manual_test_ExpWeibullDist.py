import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

from virocon.distributions import ExponentiatedWeibullDistribution



# OMAE2020_param = {"alpha" : 10.0, 
#                   "beta" : 2.42, 
#                   "delta" : 0.761
#                   }

x = np.linspace(2, 15, num=100)
p = np.linspace(0.01, 0.99, num=100)

true_alpha = 10
true_beta = 2.42
true_delta = 0.761
expweibull_samples = sts.exponweib.rvs(a=true_delta, c=true_beta, 
                                       loc=0, scale=true_alpha, 
                                       size=100, random_state=42)

# %%
# my_expweibull = ExponentiatedWeibullDistribution(**OMAE2020_param)
my_expweibull = ExponentiatedWeibullDistribution(fit_method="lsq", weights="quadratic")

my_expweibull.fit(expweibull_samples)

my_pdf = my_expweibull.pdf(x)
my_cdf = my_expweibull.cdf(x)
my_icdf = my_expweibull.icdf(p)

my_alpha = my_expweibull.alpha
my_beta = my_expweibull.beta
my_delta = my_expweibull.delta

# %%
import sys 
sys.path.append("../viroconcom")
from viroconcom.distributions import ExponentiatedWeibullDistribution

# ref_expweibull = ExponentiatedWeibullDistribution(shape=OMAE2020_param["beta"],
#                                                   scale=OMAE2020_param["alpha"],
#                                                   shape2=OMAE2020_param["delta"])

ref_expweibull = ExponentiatedWeibullDistribution()

ref_expweibull.fit(expweibull_samples)

ref_alpha = ref_expweibull.scale(None)
ref_beta = ref_expweibull.shape(None)
ref_delta = ref_expweibull.shape2(None)

ref_pdf = ref_expweibull.pdf(x)
ref_cdf = ref_expweibull.cdf(x)
ref_icdf = ref_expweibull.i_cdf(p)

# %%

# ref_data = {"ref_alpha" : ref_alpha,
#             "ref_beta" : ref_beta,
#             "ref_delta" : ref_delta,
#             "ref_pdf" : ref_pdf, 
#             "ref_cdf" : ref_cdf,
#             "ref_icdf" : ref_icdf,
#             }

# np.savez_compressed("reference_data_exp_weibull_wlsq_fit.npz", **ref_data)


# %%

plt.close("all")
plt.figure()
plt.plot(x, my_pdf, label="my pdf")
plt.plot(x, ref_pdf, label="ref pdf")
plt.plot(x, sts.exponweib.pdf(x, a=true_delta, c=true_beta, loc=0, scale=true_alpha,), label="true pdf")
plt.hist(expweibull_samples, density=True)
plt.legend()
plt.figure()
plt.plot(x, my_cdf,  label="my cdf")
plt.plot(x, ref_cdf, label="ref cdf")
plt.legend()
plt.figure()
plt.plot(p, my_icdf, label="my icdf")
plt.plot(p, ref_icdf, label="ref icdf")
plt.legend()

np.testing.assert_allclose(my_pdf, ref_pdf)
np.testing.assert_allclose(my_cdf, ref_cdf)
np.testing.assert_allclose(my_icdf, ref_icdf)


