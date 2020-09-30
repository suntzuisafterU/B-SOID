# from bsoid import *  # TODO: ???
__version__ = '0.0.1'  # TODO: HIGH: ensure version number is correct before pushing. See setup.py for version.


# user-facing io api
from bsoid.util.io import (
    read_csv,
    read_csvs,
)


# General imports for packages
from . import (
    app,
    config,
    classify,
    classify_LEGACY,
    feature_engineering,
    main_LEGACY,  # TODO: low: remove this after rewiring legacy code
    pipeline,
    streamlit,
    train,
    train_LEGACY,  # TODO: potentially remove after rewiring code
)
from .util import check_arg, io, likelihoodprocessing, bsoid_logging, statistics, videoprocessing, visuals
