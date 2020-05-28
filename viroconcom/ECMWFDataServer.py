from ecmwfapi import ECMWFDataServer


server = ECMWFDataServer(url="https://api.ecmwf.int/v1",key="",email="")


#Methodenaufruf für das Ziehen der Daten, bezogen auf das Date, Area, Time und Grid

def getData(date, time, grid, area):
    server.retrieve({
        # Specify the ERA-Interim data archive. Don't change.
        "class": "ei",
        "dataset": "interim",
        "expver": "1",
        "stream": "wave",
        # forecast (type:fc), from both daily forecast runs (time) with all available forecast steps (step, in hours)
        "type": "fc",
        "levtype": "sfc",
        # all available parameters, for codes see http://apps.ecmwf.int/codes/grib/param-db
        # 229.140/232.140 means Mean wave period/Significant height of combined wind waves and swell
        "param": "229.140/232.140",
        # days worth of data
        "date": date,
        "time": time,
        "step": "0",
        "grid": grid,
        # optionally restrict area to Europe (in N/W/S/E).
        "area": area,
        # Definition of the format
        "format": "netcdf",
        # set an output file name
        "target": "test.nc"
    })
getData("2018-06-01/to/2018-06-30", "00:00:00", "0.75/0.75", "75/-20/10/60"   )


