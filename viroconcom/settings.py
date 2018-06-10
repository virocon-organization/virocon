from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'VERSION')) as version_file:
    VERSION = version_file.read().strip()

SHAPE_STRING = "shape"
LOCATION_STRING = "loc"
SCALE_STRING = "scale"
