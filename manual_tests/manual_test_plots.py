import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from virocon import GlobalHierarchicalModel, calculate_alpha, IFORMContour
from virocon.predefined import get_DNVGL_Hs_Tz
from virocon.plotting import (plot_2D_contour, 
                              plot_2D_isodensity, 
                              plot_dependence_functions, 
                              plot_marginal_quantiles
                              )



# %% load data, prepare common variables

data = pd.read_csv("datasets/NDBC_buoy_46025.csv", sep=",")[["Hs", "T"]]


dist_descriptions, fit_descriptions, model_description = get_DNVGL_Hs_Tz()
ghm = GlobalHierarchicalModel(dist_descriptions)
ghm.fit(data, fit_descriptions=fit_descriptions)

state_duration = 3
return_period = 50 
alpha = calculate_alpha(state_duration, return_period)
iform_contour = IFORMContour(ghm, alpha)



plt.close("all")
# %% plot_qq

axes = plot_marginal_quantiles(ghm, data, model_desc=model_description)

# plt.show()

# %% plot_dependence_functions

par_rename = {"mu": r"$\mu$",
             "sigma" : r"$\sigma$"}
axes = plot_dependence_functions(ghm, model_desc=model_description, par_rename=par_rename)
    
# %% plot_isodensity

ax = plot_2D_isodensity(ghm, model_desc=model_description, sample=data, swap_axis=True)


# %% plot_contour


ax = plot_2D_contour(iform_contour, model_desc=model_description, sample=data,
                     design_conditions=True, swap_axis=True)

    
