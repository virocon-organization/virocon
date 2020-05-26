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






np.savetxt('lat.csv', lat, delimiter=',')
np.savetxt('lon.csv', lon, delimiter=',')
np.savetxt('swh.csv', lon, delimiter=',')
np.savetxt('mwp.csv', lon, delimiter=',')

df_lat = pd.DataFrame(data=lat, index=dtime)
df_lat.to_csv('lat.csv')

df_lat = pd.DataFrame(data=lon, index=dtime)
df_lat.to_csv('lon.csv')

df_lat = pd.DataFrame(data=swh, index=dtime)
df_lat.to_csv('swh.csv')

df_lat = pd.DataFrame(data=mwp, index=dtime)
df_lat.to_csv('mwp.csv')