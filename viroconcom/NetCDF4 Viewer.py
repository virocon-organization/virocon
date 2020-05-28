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

dims = nc.dimensions
dimse = len(dims)
for key in dims:
    print("dimension["+key+"] = "+str(len(dims[key])))

gattrs = nc.ncattrs()
ngattrs = len(gattrs)
print("number of global attributes ="+str(ngattrs))

for key in gattrs:
    print("global attribute["+key+"]="+str(getattr(nc,key)))

vars = nc.variables
nvars = len(vars)
print("number of variables ="+str(nvars))

for var in vars:
    print("-----------variable"+var+"------------")
    print("shape = "+str(vars[var].shape))
    vdims = vars[var].dimensions
    for vd in vdims:
        print("dimension["+vd+"]=" + str(len(dims[vd])))
var='swh'
vattrs=vars[var].ncattrs()

print("number of attributes = "+str(len(vattrs)))
for vat in vattrs:
    print("attribute["+vat+"]=" + str(getattr(vars[var],vat)))

print(var)
#slice of data
a = vars[var][1:3]
print(a)
