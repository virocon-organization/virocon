import netCDF4
import pandas as pd
import numpy as np

test_nc_file = 'test.nc'
nc = netCDF4.Dataset(test_nc_file, mode='r')

nc.variables.keys()

lat = nc.variables['latitude'][:]
lon = nc.variables['longitude'][:]
swh = nc.variables['swh'][:]
mwp = nc.variables['mwp'][:]
time_var = nc.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)






