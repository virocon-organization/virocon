import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from virocon.models import GlobalHierarchicalModel
from virocon.distributions import (WeibullDistribution, 
                                   LogNormalDistribution)
from virocon.dependencies import DependenceFunction
from virocon.contours import IFORMContour, calculate_alpha
from virocon.intervals import WidthOfIntervalSlicer
from virocon.plotting import (plot_2D_contour, 
                              plot_2D_isodensity, 
                              plot_dependence_functions, 
                              plot_marginal_quantiles
                              )



# %% load data, prepare common variables

data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]

# A 3-parameter power function (a dependence function).
def _power3(x, a, b, c):
    return a + b * x ** c

# A 3-parameter exponential function (a dependence function).
def _exp3(x, a, b, c):
    return a + b * np.exp(c * x)

bounds = [(0, None), 
          (0, None), 
          (None, None)]

power3 = DependenceFunction(_power3, bounds)
exp3 = DependenceFunction(_exp3, bounds)

dist_description_0 = {"distribution" : WeibullDistribution(),
                      "intervals" : WidthOfIntervalSlicer(width=0.5, offset=True)
                      }

dist_description_1 = {"distribution" : LogNormalDistribution(),
                      "conditional_on" : 0,
                      "parameters" : {"mu": power3,
                                      "sigma" : exp3},
                      }

ghm = GlobalHierarchicalModel([dist_description_0, dist_description_1])


ghm.fit(data)

state_duration = 3
return_period = 50 
alpha = calculate_alpha(state_duration, return_period)
iform_contour = IFORMContour(ghm, alpha)



model_description = {"names" : ["Significant wave height", "Energy wave period"],
                     "symbols" : ["H_s", "T_e"], 
                     "units" : ["m", "s"]
                     } # TODO check if correct or other wave period

plt.close("all")
# %% plot_qq

axes = plot_marginal_quantiles(ghm, data, model_desc=model_description)

# plt.show()

# %% plot_dependence_functions

par_rename = {"mu": "$\mu$",
             "sigma" : "$\sigma$"}
axes = plot_dependence_functions(ghm, model_desc=model_description, par_rename=par_rename)
    
# %% plot_isodensity

ax = plot_2D_isodensity(ghm, model_desc=model_description, sample=data, swap_axis=True)


# %% plot_contour


ax = plot_2D_contour(iform_contour, model_desc=model_description, sample=data, swap_axis=True)

    
