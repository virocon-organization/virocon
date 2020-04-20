#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "expver": "1",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "param": "167.128",
    "date": "2017-08-01/to/2017-08-03",
    "time": "00:00:00",
    "step": "3",
    "grid": "0.75/0.75",
    "area": "75/-20/10/60",
    "format": "netcdf",
    "target": "test.nc"
})