import netCDF4
import numpy as np

f = netCDF4.Dataset('C:/Users/carga/Documents/NeuestesVirocon/viroconcom/test.nc')
print(f)

swh = f.variables['swh']
print(swh)

mwp = f.variables['mwp']
print(mwp)

for d in f.dimensions.items():
    print(d)



