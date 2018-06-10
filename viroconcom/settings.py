from os import path, pardir

here = path.abspath(path.dirname(__file__))
parent_dir = path.join(here, pardir)
with open(path.join(parent_dir, 'viroconcom/VERSION')) as version_file:
    VERSION = version_file.read().strip()

SHAPE_STRING = "shape"
LOCATION_STRING = "loc"
SCALE_STRING = "scale"
