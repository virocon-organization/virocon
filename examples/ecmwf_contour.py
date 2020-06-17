import netCDF4
from viroconcom.dataECMWF import ECMWF
from matplotlib import pyplot as plt
from viroconcom.contours import DirectSamplingContour
from viroconcom.fitting import Fit
import numpy as np


# Get the sample and write them into a file.
ecmwf = ECMWF("00:00:00", "0.75/0.75", "75/-20/10/60", "229.140/232.140")
ecmwf.get_data("2017-09-30/to/2018-09-30")
# Open the file for reading.
test_nc_file = '../examples/datasets/ecmwf.nc'
nc = netCDF4.Dataset(test_nc_file, mode='r')

var_s = nc.variables
var = 'swh'
var2 = 'mwp'

# Change the data-array and get rid of nan`s.
data1 = var_s[var][:].flatten()
data2 = var_s[var2][:].flatten()

data1 = data1.filled(100.)
data2 = data2.filled(100.)
delete = np.where(data1 == 100.)

data1 = np.delete(data1, delete)
data2 = np.delete(data2, delete)
delete2 = np.where(data2 == 100.)

data1 = np.delete(data1, delete2)
data2 = np.delete(data2, delete2)
data1 = data1.round(decimals=6)
data2 = data2.round(decimals=6)


# Describe the distribution that should be fitted to the sample.
dist_description_0 = {'name': 'Weibull',
                      'dependency': (None, None, None),
                      'width_of_intervals': 2}
dist_description_1 = {'name': 'Lognormal',
                      'dependency': (0, None, 0),
                      'functions': ('exp3', None, 'power3')}
my_fit = Fit([data1, data2], [dist_description_0, dist_description_1])

dsc = DirectSamplingContour(my_fit.mul_var_dist, 5000000, 25, 24, 6)
direct_sampling_contour = dsc.direct_sampling_contour()

# Plot the contour and the sample.
fig, axes = plt.subplots(2)
axes[0].scatter(dsc.data[0], dsc.data[1], marker='.')
axes[0].plot(direct_sampling_contour[0], direct_sampling_contour[1], color='red')
axes[0].title.set_text('Monte-Carlo-Sample')
axes[0].set_ylabel('Mean wave period (s)')

axes[1].scatter(data1, data2)
axes[1].plot(direct_sampling_contour[0], direct_sampling_contour[1], color='red')
axes[1].title.set_text('Data from ECMWF')
axes[1].set_xlabel('Significant wave height (m)')
axes[1].set_ylabel('Mean wave period (s)')
plt.show()