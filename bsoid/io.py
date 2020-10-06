"""
Functions related to opening/saving files should go here
"""

from typing import Any, Collection, List, Optional, Tuple, Union
import errno
import glob
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import re
import sys
import traceback


from bsoid import config, logging_bsoid, pipeline, statistics


logger = config.initialize_logger(__name__)
ERROR_INVALID_NAME = 123  # necessary for valid filename checking. Do not remove this.


########################################################################################################################


# TODO: med/high: change function to accept either CSV or h5 files from DLC. Functionally, should be the same except for
#  deciding to use read_h5() or read_csv()
def read_csv(csv_file_path: str, **kwargs) -> pd.DataFrame:
    """
    Reads in a CSV that is assumed to be an output of DLC. The raw CSV is re-formatted to be more
    friendly towards data manipulation later in the B-SOiD feature engineering pipeline.
        * NO MATH IS DONE HERE, NO DATA IS REMOVED *

    :param csv_file_path: (str, absolute path) The input file path requires the CSV file in question to be
        an output of the DLC process. If the file is not, use pd.read_csv() instead.

    EXAMPLE data: DataFrame directly after invoking pd.read_csv(csv, header=None):
                   0                                              1                                               2                                               3                                                4  ...
        0     scorer  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  ...
        1  bodyparts                                      Snout/Head                                      Snout/Head                                      Snout/Head                               Forepaw/Shoulder1  ...
        2     coords                                               x                                               y                                      likelihood                                               x  ...
        3          0                                 1013.7373046875                                661.953857421875                                             1.0                              1020.1138305664062  ...
        4          1                              1012.7627563476562                               660.2426147460938                                             1.0                              1020.0912475585938  ...

    :return: (DataFrame)
        EXAMPLE OUTPUT:
                 Snout/Head_x       Snout/Head_y Snout/Head_likelihood Forepaw/Shoulder1_x Forepaw/Shoulder1_y Forepaw/Shoulder1_likelihood  ...                                          scorer
        0     1013.7373046875   661.953857421875                   1.0  1020.1138305664062   621.7146606445312           0.9999985694885254  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        1  1012.7627563476562  660.2426147460938                   1.0  1020.0912475585938   622.9310913085938           0.9999995231628418  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        2  1012.5982666015625   660.308349609375                   1.0  1020.1837768554688   623.5087280273438           0.9999994039535522  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        3  1013.2752685546875  661.3504028320312                   1.0     1020.6982421875   624.2875366210938           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
        4  1013.4093017578125  661.3643188476562                   1.0  1020.6074829101562     624.48486328125           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
    """
    # Arg checking
    if not os.path.isfile(csv_file_path):
        err = f'Input filepath to csv was not a valid file path: {csv_file_path} (type = {type(csv_file_path)}.'
        logger.error(err)
        raise ValueError(err)
    # Read in kwargs
    nrows = kwargs.get('nrows', sys.maxsize)  # TODO: address case where nrows is <= 3 (no data parsed then)
    filename = csv_file_path  # os.path.split(csv_file_path)[-1]
    file_without_suffix = filename[:filename.rfind('.')]
    # # # # # # #
    # Read in CSV
    df = pd.read_csv(csv_file_path, header=None, nrows=nrows)
    # # Manipulate Frame based on top row
    # Check that the top row is like [scorer, DLCModel, DLCModel.1, DLCModel.2, ...] OR [scorer, DLCModel, DLCModel,...]
    # Use regex to truncate the decimal/number suffix if present.
    top_row_values_set: set = set([re.sub(r'(.*)(\.\w+)?', r'\1', x) for x in df.iloc[0]])
    top_row_without_scorer: tuple = tuple(top_row_values_set - {'scorer', })
    if len(top_row_without_scorer) != 1:
        non_standard_dlc_top_row_err = f'The top row of this DLC file ({csv_file_path}) is not standard. ' \
                                       f'Top row values set = {top_row_values_set}. / ' \
                                       f'DataFrame = {df.head().to_string()}'
        logger.error(non_standard_dlc_top_row_err)
        raise ValueError(non_standard_dlc_top_row_err)
    # Save scorer/model name for later column creation
    dlc_scorer = top_row_without_scorer[0]
    # Remove top row (e.g.: [scorer, DLCModel, DLCModel, ... ]) now that we have saved the model name
    df = df.iloc[1:, :]

    # # Manipulate Frame based on next two rows to create column names.
    # Create columns based on next two rows. Combine the tow rows of each column and separate with "_"
    array_of_next_two_rows = np.array(df.iloc[:2, :])
    new_column_names: List[str] = []
    for col_idx in range(array_of_next_two_rows.shape[1]):
        new_col = f'{array_of_next_two_rows[0, col_idx]}_{array_of_next_two_rows[1, col_idx]}'
        new_column_names += [new_col, ]
    df.columns = new_column_names

    # Remove next two rows (just column names, no data here) now that columns names are instantiated
    df = df.iloc[2:, :]

    # # Final touches
    # Delete "coords" column since it is just a numerical counting of rows. Not useful data.
    df = df.drop('bodyparts_coords', axis=1)
    # Convert all values to float in case they are parsed as string
    df = df.astype(np.float)
    # Reset index (good practice) after chopping off top 3 columns so index starts at 0 again
    df = df.reset_index(drop=True)
    # Instantiate 'scorer' column so we can track the model if needed later
    df['scorer'] = dlc_scorer
    # Save source for future use
    df['source'] = filename
    # Number the frames
    df['frame'] = range(len(df))

    return df


def read_pipeline(path_to_file: str) -> pipeline:
    """
    With a valid path, read in an existing pipeline
    :param path_to_file:
    :return:
    """
    # TODO: do final checks on this funct
    if not os.path.isfile(path_to_file):
        invalid_file_err = f'Invalid path to pipeline: {path_to_file}. Cannot open file. '
        logger.error(invalid_file_err)
        raise ValueError(invalid_file_err)
    with open(path_to_file, 'rb') as file:
        p = joblib.load(file)
    return p


########################################################################################################################

def check_folder_for_dlc_output_files(folder_path: str, file_extension: str,
                                      check_recursively: bool = False, sort_names: bool = False) -> List[str]:
    """

    :param file_extension:
    :param folder_path:
    :param check_recursively:
    :param sort_names:
    :return:
    """
    if not os.path.isdir(folder_path):
        value_err = f'This path is not a valid path to a folder: {folder_path} ' \
                    f'(type = {type(folder_path)}).'
        logger.error(value_err)
        raise ValueError(value_err)
    folder_path = os.path.abspath(folder_path)
    if check_recursively:
        path_to_check_for_csvs = f'{folder_path}{os.path.sep}**{os.path.sep}*.{file_extension}'
    else:
        path_to_check_for_csvs = f'{folder_path}{os.path.sep}*.{file_extension}'

    logger.debug(f'{logging_bsoid.get_current_function()}: Path that is being '
                 f'checked using glob selection: {path_to_check_for_csvs}')

    file_names: List[str] = glob.glob(path_to_check_for_csvs, recursive=check_recursively)

    if sort_names: statistics.sort_list_nicely_in_place(file_names)

    return file_names


###

def has_invalid_chars_in_name_for_a_file(name, additional_characters: Optional[Collection[str]] = None) -> bool:
    """
    Checks if an invalid characters have been included in a potential path. Useful for checking user
    input before attempting to save files. The list of invalid characters
    :param name:
    :param additional_characters:
    :return:
    """
    if additional_characters is not None \
            and not isinstance(additional_characters, list) \
            and not isinstance(additional_characters, tuple) \
            and not isinstance(additional_characters, set):
        invalid_type_err = f'{inspect.stack()[0][3]}(): Invalid type ' \
                       f'found: {type(additional_characters)} (value: {additional_characters})'
        logger.error(invalid_type_err)
        raise TypeError(invalid_type_err)

    invalid_chars_for_windows_files = {':', '*', '\\', '/', '?', '"', '<', '>', '|'}
    if additional_characters is not None:
        invalid_chars_for_windows_files = invalid_chars_for_windows_files.union(set(additional_characters))
    if not isinstance(name, str) or not name:
        return True
    union_of_string_and_invalid_chars = set(name).intersection(invalid_chars_for_windows_files)
    if len(union_of_string_and_invalid_chars) != 0:
        logger.error(f'Union = {union_of_string_and_invalid_chars}')
        return True

    return False


def is_pathname_valid(pathname: str) -> bool:
    """ Checks if the path name is valid. Useful for checking user inputs.
    Source: https://stackoverflow.com/a/34102855
    Returns: (bool) `True` if the passed pathname is a valid pathname for the current OS;
                    `False` otherwise.
    """
    # If this pathname is either not a string or is but is empty, this pathname is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this path name's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


if __name__ == '__main__':
    path = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\' \
           f'sample_train_data_folder\\Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
    folder = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\sample_train_data_folder\\'
    # read_csv(path)
    x = check_folder_for_dlc_output_files(folder, 'csv', check_recursively=False)
    print(x)
    print(len(x))
    pass

