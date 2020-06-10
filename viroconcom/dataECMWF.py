"""
You must follow the steps on: https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets
If you will face some problems, ecmwf provides reasons and solutions on their website for any problem.
Here is a short list of things you will need to do for this script to work:
    1. Make an account on ecmwf.
    2. Make your credentials to environment variables on your computer. This is important since the api will check them,
        see: https://confluence.ecmwf.int/display/WEBAPI/How+to+retrieve+ECMWF+Public+Datasets
        Note: You have to be logged in on ecmwf to see how to set environment variables
    3. Install package ecmwf-api-client recommended using pip
    4. There is a certificate in viroconcom the file is named: quovadis_rca2g3_der.cer
        Install it.
    5. Now the code should run. If you have further troubles see ecmwf website for troubleshooting.
"""

from ecmwfapi import ECMWFDataServer
import numpy as np

# For this to work, you must have an ecmwf account and put your key and E-Mail in this form.
server = ECMWFDataServer(url="https://api.ecmwf.int/v1",
                         key="", email="")


class ECMWF():
    """
    Class with method to retrieve ecmwf data from server.

    Attributes
    ----------
    None.

    """

    def get_data(self, date, time, grid, area, param):
        """
        Gets specific data.

        Parameters
        ----------
        date : str
            Form: "year-month-day/to/year-month-day"
            You will get the data for this time period.
        time : str
            Form: "00:00:00"
            This gives you one measurement per day.
        grid : str
            Form: "0.75/0.75"
            How small you want the space between measuring points.
        area : str
            Form: "75/-20/10/60"
            Which area you data should come from.
        param : str
            Form: "229.140/232.140"
            There are different codes for different parameter. See: http://apps.ecmwf.int/codes/grib/param-db
            These: "229.140/232.140" are for mean wave period / significant height of combined wind, waves and swell

        """
        server.retrieve({
            # Specify the ERA-Interim data archive. Don't change.
            "class": "ei",
            "dataset": "interim",
            "expver": "1",
            "stream": "wave",
            # Forecast (type:fc), from both daily forecast runs (time)
            # with all available forecast steps (step, in hours).
            "type": "fc",
            "levtype": "sfc",
            # All available parameters, for codes see http://apps.ecmwf.int/codes/grib/param-db .
            # 229.140/232.140 means mean wave period / significant height of combined wind waves and swell.
            "param": param,
            # Days worth of data.
            "date": date,
            "time": time,
            "step": "0",
            "grid": grid,
            # Specify as North/West/South/East in Geographic lat/long degrees.
            # Southern latitudes and Western longitudes must be given as negative numbers.
            "area": area,
            # Definition of the format.
            "format": "netcdf",
            # Set an output file name.
            "target": '../examples/datasets/ecmwf.nc'
        })
