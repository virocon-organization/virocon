import numpy as np
import pandas as pd

from virocon import (DependenceFunction, WeibullDistribution, 
                     WidthOfIntervalSlicer, LogNormalDistribution, 
                     GlobalHierarchicalModel, calculate_alpha, IFORMContour)

from virocon.utils import calculate_design_conditions



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
                      "intervals" : WidthOfIntervalSlicer(width=0.5)
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



# %%
import matplotlib.pyplot as plt







plt.close()
coords = iform_contour.coordinates
x1 = np.append(coords[:, 1], coords[0, 1])
y1 = np.append(coords[:, 0], coords[0, 0])
plt.plot(x1, y1, c="#BB5566")
design_conditions = calculate_design_conditions(iform_contour, swap_axis=True)
plt.scatter(design_conditions[:, 0], design_conditions[:, 1], c="#004488", marker="x", 
            zorder=2.5)
my_steps = np.arange(3, 23, 0.5)
design_conditions2 = calculate_design_conditions(iform_contour, steps=my_steps, swap_axis=True)
plt.scatter(design_conditions2[:, 0], design_conditions2[:, 1], c="#DDAA33", marker="x", 
            zorder=2.5)
plt.show()
    

