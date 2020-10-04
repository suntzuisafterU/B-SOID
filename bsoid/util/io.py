"""
Functions related to opening/saving files should go here
"""

from typing import Any, List, Tuple
import errno
import glob
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import re
import sys


from bsoid import config, feature_engineering, pipeline
from bsoid.util import likelihoodprocessing
from bsoid.util.bsoid_logging import get_current_function

logger = config.initialize_logger(__name__)
ERROR_INVALID_NAME = 123  # necessary for valid filename checking


########################################################################################################################


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
    nrows = kwargs.get('nrows', sys.maxsize)
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


def read_pipeline(path_to_file: str) -> pipeline.Pipeline:
    """
    With a valid path, read in an existing pipeline
    :param path_to_file:
    :return:
    """
    # TODO: do final checks on this funct
    if not os.path.isfile(path_to_file):
        invalid_file_err = f'Invalid path: {path_to_file}. Cannot open, is not file.'
        logger.error(invalid_file_err)
        raise ValueError(invalid_file_err)
    with open(path_to_file, 'rb') as file:
        p = joblib.load(file)
    return p


########################################################################################################################

def check_for_csv_files_in_path(folder_path: str, check_recursively=False) -> List[str]:
    """

    :param folder_path: (str) an absolute path to a folder which will be checked by this function
    :param check_recursively: (bool) indicates whether or not the folder should be
        checked recursively into ALL subfolders or not checked recursively at all.
    :return: (List[str]) A list of absolute paths to csvs
    """
    # Arg checking
    if not os.path.isdir(folder_path) and os.path.abspath(folder_path):
        folder_err = f'Invalid folder specified. Check that it exists and that'
        logger.error(folder_err)
        raise ValueError(folder_err)
    #
    path_to_check_for_csvs = f'{folder_path}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'{get_current_function()}: Path that is being checked using glob selection: {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=check_recursively)
    likelihoodprocessing.sort_list_nicely_in_place(filenames)
    logger.info(f'{get_current_function()}: Total files found: {len(filenames)}. List of files found: {filenames}.')
    if False in [os.path.abspath(folder) for folder in filenames]:
        raise ValueError(f'FOLDERS NOT ABS PATHS: ', filenames)

    return filenames


def read_csvs(*sources) -> List[pd.DataFrame]:
    """
    Give a source/sources of .csv file, return a list of Pandas DataFrames
    :param source: (valid types: str, List[str], or Tuple[str]) sources of .csv files to read in. These
        csv files are expected to be of DLC output after video analysis. The general layout format
        expected is as follows:

    :return:
    """
    raise NotImplementedError(' Development put on hold. Not yet implemented.')
    # Check args
    if isinstance(source, str):
        sources = [source, ]
    elif isinstance(source, list) or isinstance(source, tuple):
        sources = source
    else:
        type_err = f'An invalid type was detected. Type was expected to be in {{str, List[str], Tuple[str]}}'
        logger.error(type_err)
        raise TypeError(type_err)

    # Resolve csv file paths
    # Read in csv files and return
    list_df: List[pd.DataFrame] = []
    return list_df


def check_folder_contents_for_csv_files(absolute_folder_path: str, check_recursively: bool = False) -> List[str]:
    """ Legacy? TODO: review this func
    Finished.
    # TODO: purpose
    :param absolute_folder_path: (str) an absolute path, TODO
    :param check_recursively: (bool) TODO
    :return: TODO
        Returns List of absolute paths to CSVs detected
    """
    # Check args
    if not os.path.isdir(absolute_folder_path):
        value_err = f'This path is not a valid path to a folder: {absolute_folder_path} ' \
                    f'(type = {type(absolute_folder_path)}).'
        logger.error(value_err)
        raise ValueError(value_err)
    # Continue if values valid
    path_to_check_for_csvs = f'{absolute_folder_path}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'{get_current_function()}: Path that is being checked using glob selection: {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=check_recursively)
    likelihoodprocessing.sort_list_nicely_in_place(filenames)
    logger.info(f'{get_current_function()}: Total files found: {len(filenames)}. List of files found: {filenames}.')
    return filenames


### Legacy funcs kept for continuity

def get_videos_from_folder_in_BASEPATH(folder_name: str, video_extension: str = 'mp4') -> List[str]:
    """ * Legacy *
    Previously named `get_video_names()`
    Gets a list of video files within a folder
    :param folder_name: str, folder path. Must reside in BASE_PATH.
    :param video_extension:
    :return: (List[str]) video file names all of which have a .mp4 extension
    """
    if not isinstance(folder_name, str):
        err = f'`{inspect.stack()[0][3]}(): folder_name was expected to be of ' \
              f'type str but instead found {type(folder_name)}.'
        logger.error(err)
        raise TypeError(err)
    path_to_folder = os.path.join(config.DLC_PROJECT_PATH, folder_name)

    path_to_folder_with_glob = f'{path_to_folder}/*.{video_extension}'
    logger.debug(f'{inspect.stack()[0][3]}(): Path to check for videos: {path_to_folder_with_glob}.')
    video_names = glob.glob(path_to_folder_with_glob)
    likelihoodprocessing.sort_list_nicely_in_place(video_names)
    return video_names


def get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder: str) -> List:
    """
    Get_filenames() makes the assumption that the folder is in PROJECT Path; however, it is an obfuscated assumption
    and bad. A new function that DOES NOT RESOLVE PATH IMPLICITLY WITHIN should be created and used.
    :param folder:
    :return:
    """
    path_to_check_for_csvs = f'{config.DLC_PROJECT_PATH}{os.path.sep}{folder}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'{get_current_function()}():Path that is being checked using glob selection:{path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=True)
    likelihoodprocessing.sort_list_nicely_in_place(filenames)
    logger.info(f'{get_current_function()}(): Total files found: {len(filenames)}. List of files found: {filenames}.')
    return filenames


def import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(folders_in_project_path: list) -> Tuple[List[List[str]], List[np.ndarray], List]:
    """
    Import multiple folders containing .csv files and process them
    :param folders_in_project_path: List[str]: Data folders
    :return filenames: list, data filenames
    :return data: List of arrays, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    # TODO: what does `raw_data_list` do? It looks like a variable without a purpose. It appends but does not return.

    if len(folders_in_project_path) == 0:
        empty_folders_list_err = f'{inspect.stack()[0][3]}: argument `folders` list is empty. No folders to check.'
        logger.error(empty_folders_list_err)
        raise ValueError(empty_folders_list_err)

    file_names_list, raw_data_list, list_of_arrays_of_data, perc_rect_list = [], [], [], []
    # Iterate over folders
    for idx_folder, folder in enumerate(folders_in_project_path):  # Loop through folders
        filenames_found_in_current_folder: List[str] = get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder)
        for idx_filename, filename in enumerate(filenames_found_in_current_folder):
            logger.debug(f'{get_current_function()}(): Importing CSV file #{idx_filename+1}, {filename}, from folder #{idx_folder+1}')
            df_current_file_data = pd.read_csv(filename, low_memory=False)
            array_current_file_data_adaptively_filtered, perc_rect = feature_engineering.process_raw_data_and_filter_adaptively(df_current_file_data)
            logger.debug(f'{get_current_function()}(): Done preprocessing (x,y) from file #{idx_filename+1}, folder #{idx_folder+1}.')
            raw_data_list.append(df_current_file_data)
            perc_rect_list.append(perc_rect)
            list_of_arrays_of_data.append(array_current_file_data_adaptively_filtered)
        file_names_list.append(filenames_found_in_current_folder)
        logger.debug(f'{get_current_function()}(): Processed {len(filenames_found_in_current_folder)} CSV files from folder: {folder}')
    # array_of_arrays_of_data: np.ndarray = np.array(data_list)
    logger.info(f'{get_current_function()}(): Processed a total of {len(list_of_arrays_of_data)} CSV files')  # and compiled into a {array_of_arrays_of_data.shape} data list/array.')
    return file_names_list, list_of_arrays_of_data, perc_rect_list


def import_folders_app(ost_project_path, input_folders_list: list, BODYPARTS: dict) -> Tuple[List, List, np.ndarray, List]:
    """ the _app version of import folders """
    warning = f'Change usage from lilkelihoodprocessing to io. Caller = {inspect.stack()[1][3]}'
    logger.warning(warning)
    return import_folders_app(ost_project_path, input_folders_list, BODYPARTS)


###

def has_invalid_chars_in_name_for_a_file(name) -> bool:
    invalid_chars_for_windows_files = {':', '*', '\\', '/', '?', '"', '<', '>', '|'}
    if not isinstance(name, str) or not name:
        return True
    union = set(name).intersection(invalid_chars_for_windows_files)
    if len(union) != 0:
        logger.error(f'Union = {union}')
        return True

    return False


def is_pathname_valid(pathname: str) -> bool:
    """
    Source: https://stackoverflow.com/a/34102855
    `True` if the passed pathname is a valid pathname for the current OS;
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
    path = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\sample_train_data_folder\\Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
    read_csv(path)
    pass

