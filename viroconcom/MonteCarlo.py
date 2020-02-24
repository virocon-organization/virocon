# data: WSPD APD
#       xyz  xyz
#       ...  ...
# WSPD: WindSpeed
# WVHT: significantWaveHight
# APD: wavePeriode
#
# Hier erstmal nach dem Paper von elsevier Ocean Engineering
#
# Variablen:
# Tz = wavePeriod
# Hs = significantWaveHeight


#
# Daten aufbereiten.
#

import csv
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from scipy import integrate
from viroconcom.viroconcom.distributions import WeibullDistribution
param = WeibullDistribution
sample_1 = WeibullDistribution.calculate_data(param, 10000, shape=1.6, loc=0.9, scale=2.8)
print(sample_1)