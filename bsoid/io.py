"""
Functions related to opening/saving files should go here
"""

from typing import Any, Collection, List, Optional, Tuple, Union
import glob
import joblib
import numpy as np
import os
import pandas as pd
import re
import sys


from bsoid import check_arg, config, logging_bsoid, pipeline, statistics


logger = config.initialize_logger(__name__)


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
    file_path = csv_file_path  # os.path.split(csv_file_path)[-1]
    file_folder, file_name = os.path.split(file_path)
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
    # File source __________
    df['file_source'] = file_path
    # Save data file name (different from pathing source)
    df['data_source'] = file_name
    # Number the frames
    df['frame'] = range(len(df))

    # Save source for future use
    # df['source'] = file_path

    return df


def read_h5(data_file_path, **kwargs) -> pd.DataFrame:
    # TODO: HIGH IMPLEMENT! :)
    raise NotImplementedError(f'')
    return


def read_dlc_data(data_file_path: str, **kwargs) -> pd.DataFrame:
    """

    :param data_file_path:
    :param kwargs:
    :return:
    """
    check_arg.ensure_is_file(data_file_path)
    file_extension = data_file_path.split('.')[-1]

    if file_extension == 'csv':
        return read_csv(data_file_path)
    elif file_extension == 'h5':
        return read_h5(data_file_path)
    else:
        invalid_ext_err = f'TODO: ELABORATE: invalid file trying to be read-in as DLC output'
        logger.error(invalid_ext_err)
        raise ValueError(invalid_ext_err)


def read_pipeline(path_to_file: str) -> pipeline.BasePipeline:
    """
    With a valid path, read in an existing pipeline
    :param path_to_file:
    :return:
    """
    # TODO: do final checks on this function
    check_arg.ensure_is_file(path_to_file)
    with open(path_to_file, 'rb') as file:
        p = joblib.load(file)

    return p


def save_pipeline(pipeline_obj, path_to_folder: str = config.OUTPUT_PATH) -> pipeline.BasePipeline:
    """
    With a valid path, save an existing pipeline
    :param pipeline_obj:
    :param path_to_folder:
    :return:
    """
    # TODO: do final checks on this function   <---------------------------------------------------------------------------
    # Arg checking
    check_arg.ensure_type(path_to_folder, str)
    # check_arg.ensure_is_valid_path(path_to_folder)

    final_out_path = os.path.join(
        path_to_folder,
        pipeline.generate_pipeline_filename_from_pipeline(pipeline_obj)
    )
    with open(final_out_path, 'wb') as file:
        joblib.dump(pipeline_obj, file)

    return read_pipeline(final_out_path)


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


# Legacy Functions

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
    logger.debug(f'{logging_bsoid.get_current_function()}: Path that is being checked using glob selection: {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=check_recursively)
    statistics.sort_list_nicely_in_place(filenames)
    logger.info(f'{logging_bsoid.get_current_function()}: Total files found: {len(filenames)}. List of files found: {filenames}.')
    return filenames


def get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder: str) -> List:
    """
    Get_filenames() makes the assumption that the folder is in PROJECT Path; however, it is an obfuscated assumption
    and bad. A new function that DOES NOT RESOLVE PATH IMPLICITLY WITHIN should be created and used.
    :param folder:
    :return:
    """
    path_to_check_for_csvs = f'{config.DLC_PROJECT_PATH}{os.path.sep}{folder}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'{logging_bsoid.get_current_function()}():Path that is being checked using glob selection:{path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=True)
    statistics.sort_list_nicely_in_place(filenames)
    logger.info(f'{logging_bsoid.get_current_function()}(): Total files found: {len(filenames)}. List of files found: {filenames}.')
    return filenames


# Debugging efforts

if __name__ == '__main__':
    path = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\' \
           f'sample_train_data_folder\\Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
    folder = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\sample_train_data_folder\\'
    # read_csv(path)
    x = check_folder_for_dlc_output_files(folder, 'csv', check_recursively=False)
    print(x)
    print(len(x))
    pass

