
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer(url="https://api.ecmwf.int/v1",key="5779c25ba27168b4e35275198c308319",email="lbekov@uni-bremen.de")



# To run this example, you need an API key
# available from https://api.ecmwf.int/v1/key/

server = ECMWFDataServer()
server.retrieve({
    'origin'    : "ecmf",
    'levtype'   : "sfc",
    'number'    : "1",
    'expver'    : "prod",
    'dataset'   : "tigge",
    'step'      : "0/6/12/18",
    'area'      : "70/-130/30/-60",
    'grid'      : "2/2",
    'param'     : "167",
    'time'      : "00/12",
    'date'      : "2014-11-01",
    'type'      : "pf",
    'class'     : "ti",
    'target'    : "tigge_2014-11-01_0012.grib"
})
