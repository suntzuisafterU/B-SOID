# from bsoid import *  # TODO: ???
__version__ = '0.0.1'  # TODO: HIGH: ensure version number is correct before pushing. See setup.py for version.

# General imports for packages
from . import (
    app,
    config,
    classify,
    classify_LEGACY,
    feature_engineering,
    main_LEGACY,  # TODO: low: remove this after rewiring legacy code (?)
    pipeline,
    bsoid_streamlit,
    train,
    train_LEGACY,  # TODO: potentially remove after rewiring code
    util,
)

# user-facing io api
from .util.io import (
    read_csv,
    read_csvs,
    read_pipeline,
)

# from .util import io
# from .util import check_arg, io, likelihoodprocessing, bsoid_logging, statistics, videoprocessing, visuals
