"""

"""
from typing import Collection
import pandas as pd

from bsoid.logging_bsoid import get_caller_function
import bsoid

logger = bsoid.config.initialize_logger(__file__)


###

def ensure_type(var, expected_type):
    """"""
    if not isinstance(var, expected_type):
        type_err = f'Caller: {get_caller_function()}(): For object (value = {var}), ' \
                   f'expected type was {expected_type} but instead found {type(var)}'
        logger.error(type_err)
        raise TypeError(type_err)


def ensure_collection_not_empty(collection):
    """"""
    if len(collection) == 0:
        err = f'Caller: {get_caller_function()}(): Input variable was expected ' \
              f'to be non-empty but was in fact empty. Value = {collection}.'
        logger.error(err)
        raise ValueError(err)


def ensure_columns_in_DataFrame(columns: Collection[str], df: pd.DataFrame) -> None:
    ensure_type(df, pd.DataFrame)
    set_df_columns = set(df.columns)
    for col in columns:
        if col not in set_df_columns:
            err = f'Caller: {get_caller_function()}(): column named `{col}` was expected to be in DataFrame ' \
                  f'columns but was not found. Actual columns found: {df.columns}.'
            logger.error(err)
            raise ValueError(err)


