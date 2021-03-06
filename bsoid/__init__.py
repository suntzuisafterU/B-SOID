"""
TODO: med: Add in description of [this package]
"""

__version__ = '0.0.1'  # TODO: HIGH: ensure version number is correct before pushing. See setup.py for version.

# General imports for packages
from . import (
    app,
    check_arg,
    config,
    classify,
    feature_engineering,
    io,
    logging_bsoid,
    main_LEGACY,  # TODO: low: remove this after rewiring legacy code (?)
    pipeline,
    statistics,
    streamlit_bsoid,
    streamlit_session_state,
    # train,
    util,
    videoprocessing,
    visuals,
)

# # User-facing io api
from .io import (  # TODO: change to "from .io ..."
    read_csv,
    read_dlc_data,
    read_pipeline,
    save_pipeline,
)
