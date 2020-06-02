"""
You must follow the steps on: https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets
You will probably face some problems, but ecmwf provides reasons and solutions on their website for any problem.
Here is a short list of things you will need to do for this script to work:
    1. Make an account on ecmwf.
    2. Save your api credentials. You can get them from ecmwf website: https://api.ecmwf.int/v1/key/
    3. Make them to environment variables on your computer. This is important since the api will check them,
        see: https://confluence.ecmwf.int/display/WEBAPI/How+to+retrieve+ECMWF+Public+Datasets
        Note: You have to be logged in to see how to set environment variables
    4. Install package ecmwf-api-client recommended using pip
    5. Now you can run the code, after the 10th try, it will give you an Error.
        To solve this you have to do the following steps.
    6. There is a certificate in viroconcom the file is named: quovadis_rca2g3_der.cer
        Install it.
    7. Run the Code again. it will fail and gives you a link, You have to accept the agreement.
    8. Now the code should run. If you have further troubles see ecmwf website for troubleshooting.

    Note: Go to viroconcom/dataECMWF.py to put your credentials in the server form.
"""

import netCDF4
from viroconcom.dataECMWF import ECMWF
from matplotlib import pyplot as plt


# Get the data and write them into a file.
ECMWF.get_data(ECMWF, "2018-09-01/to/2018-09-30", "00:00:00", "0.75/0.75", "75/-20/10/60", "229.140/232.140")
# Open the file for reading.
test_nc_file = '../examples/ecmwf.nc'
nc = netCDF4.Dataset(test_nc_file, mode='r')
# Print the Dimensions.
dims = nc.dimensions
for key in dims:
    print("dimension: ["+key+"] = "+str(len(dims[key])))
# Print number of global attributes.
glob_attrs = nc.ncattrs()
##################################num_gattrs = len(gattrs)
print("Number of global attributes = "+str(len(glob_attrs)))
# Print global attributes.
for key in glob_attrs:
    print("Global attribute: ["+key+"]= "+str(getattr(nc, key)))
# Print number of variables.
var_s = nc.variables
####################################nvars = len(vars)
print("Number of variables = "+str(len(var_s)))
# Print which variables are available.
for var in var_s:
    print("--------Variable "+var+"--------")
    print("Shape = "+str(var_s[var].shape))
    var_dims = var_s[var].dimensions
    for vd in var_dims:
        print("Dimension ["+vd+"]= " + str(len(dims[vd])))

# Now, if you want to print a slice of the data, choose the variable, here we choose 'swh'
# which means significant height of combined wind waves and swell.

# First print attributes.
var = 'swh'
var_attrs = var_s[var].ncattrs()
print("Number of attributes = "+str(len(var_attrs)))
for vat in var_attrs:
    print("Attribute ["+vat+"]= " + str(getattr(var_s[var], vat)))
# Now print the slice of data.
data = var_s[var][1:2]
print(data)
# And plot part of it against 'mwp' data.
plt.scatter(data[0], var_s['mwp'][1:2][0], marker='.')
plt.xlabel('significant height of combined wind waves and swell (m)')
plt.ylabel('mean wave period (s)')
plt.show()

