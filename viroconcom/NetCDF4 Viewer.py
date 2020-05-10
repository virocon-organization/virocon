import netCDF4
import pandas as pd
import matplotlib.pyplot as plt


# NetCDF4-Python can read a remote OPeNDAP dataset or a local NetCDF file:
url='http://thredds.ucar.edu/thredds/dodsC/grib/NCEP/WW3/Global/Best'
nc = netCDF4.Dataset(url)
nc.variables.keys()

lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
time_var = nc.variables['time']
dtime = netCDF4.num2date(time_var[:],time_var.units)

# determine what longitude convention is being used [-180,180], [0,360]
print lon.min(),lon.max()

# specify some location to extract time series
lati = 41.4; loni = -67.8 +360.0  # Georges Bank

# find closest index to specified value
def near(array,value):
    idx=(abs(array-value)).argmin()
    return idx

# Find nearest point to desired location (could also interpolate, but more work)
ix = near(lon, loni)
iy = near(lat, lati)

# Extract desired times.
# 1. Select -+some days around the current time:
start = dt.datetime.utcnow()- dt.timedelta(days=3)
stop = dt.datetime.utcnow()+ dt.timedelta(days=3)
#       OR
# 2. Specify the exact time period you want:
#start = dt.datetime(2013,6,2,0,0,0)
#stop = dt.datetime(2013,6,3,0,0,0)

istart = netCDF4.date2index(start,time_var,select='nearest')
istop = netCDF4.date2index(stop,time_var,select='nearest')
print istart,istop

# Get all time records of variable [vname] at indices [iy,ix]
vname = 'Significant_height_of_wind_waves_surface'
#vname = 'surf_el'
var = nc.variables[vname]
hs = var[istart:istop,iy,ix]
tim = dtime[istart:istop]

# Create Pandas time series object
ts = pd.Series(hs,index=tim,name=vname)

# Use Pandas time series plot method
ts.plot(figsize(12,4),
   title='Location: Lon=%.2f, Lat=%.2f' % ( lon[ix], lat[iy]),legend=True)
plt.ylabel(var.units);

#write to a CSV file
ts.to_csv('time_series_from_netcdf.csv')


