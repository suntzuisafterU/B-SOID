# from bsoid import *  # TODO: ???
__version__ = '0.0.1'  # TODO: HIGH: ensure version number is correct before pushing. See setup.py for version.


# user-facing io api
from bsoid.util.io import (
    read_dlc_csv,
    read_csvs,
)


# General imports for packages
from . import app, config, classify, feature_engineering, train
from .util import check_arg, io, likelihoodprocessing, bsoid_logging, statistics, videoprocessing, visuals

