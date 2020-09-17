"""
likelihood processing utilities
Forward fill low likelihood (x,y)
"""

from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import glob
import inspect
import numpy as np
import os
import pandas as pd
import re

from bsoid import config


logger = config.initialize_logger(__name__)


########################################################################################################################
def get_current_function() -> str:
    """"""
    return inspect.stack()[1][3]


def boxcar_center(input_array, n) -> np.ndarray:
    """
    TODO
    :param input_array: TODO
    :param n: TODO
    :return: TODO
    """
    input_array_as_series = pd.Series(input_array)
    moving_avg = np.array(input_array_as_series.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg


def convert_int_from_string_if_possible(s: str):
    """ Converts digit string to integer """
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s) -> List:
    """
    Turn a string into a list of string and number chunks.
        e.g.: input: "z23a" -> output: ["z", 23, "a"]
    """
    return [convert_int_from_string_if_possible(c) for c in re.split('([0-9]+)', s)]


def sort_list_nicely_in_place(list_input: list) -> None:
    """ Sort the given list (in place) in the way that humans expect. """
    if not isinstance(list_input, list):
        raise TypeError(f'argument `l` expected to be of type list but '
                        f'instead found: {type(list_input)} (value: {list_input}).')
    list_input.sort(key=alphanum_key)


########################################################################################################################

def get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder: str) -> List:
    """
    Get_filenames() makes the assumption that the folder is in PROJECT Path; however, it is an obfuscated assumption
    and bad. A new function that DOES NOT RESOLVE PATH IMPLICITLY WITHIN should be created and used.
    :param folder:
    :return:
    """
    path_to_check_for_csvs = f'{config.DLC_PROJECT_PATH}{os.path.sep}{folder}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'{get_current_function()}: Path that is being checked using glob selection: {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=True)
    sort_list_nicely_in_place(filenames)
    logger.info(f'{get_current_function()}: Total files found: {len(filenames)}. List of files found: {filenames}.')
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
            array_current_file_data_adaptively_filtered, perc_rect = process_raw_data_and_filter_adaptively(df_current_file_data)
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
    """ the _app version of import folders
    Import multiple folders containing .csv files and process them
    :param input_folders_list: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    # TODO: the _app implementation of import_folders(); needs to be reconciled with non-_app implementation but tuple reponse length does not match
    fldrs, data_list, perc_rect_list = [], [], []
    all_file_names_list = []
    for idx_folder, folder in enumerate(input_folders_list):  # Loop through folders
        file_names_in_current_folder = get_filenames_csvs_from_folders_recursively_in_dlc_project_path(ost_project_path, folder)  # TODO: HIGH: resolve args <-------------------------------------
        for idx_filename, filename in enumerate(file_names_in_current_folder):
            logger.debug(f'{get_current_function()}(): Importing CSV file {idx_filename+1} from folder {idx_folder+1}')
            df_file_i = pd.read_csv(filename, low_memory=False)
            df_file_i_filtered, perc_rect = adaptive_filter_data_app(df_file_i, BODYPARTS)  # curr_df_filt, perc_rect = adaptive_filter_data_app(df_file_i, BODYPARTS)
            logger.debug(f'{get_current_function()}(): Done preprocessing (x,y) from file {idx_filename+1}, folder {idx_folder+1}.')
            # rawdata_li.append(curr_df)
            perc_rect_list.append(perc_rect)
            data_list.append(df_file_i_filtered)
        fldrs.append(folder)
        all_file_names_list.append(file_names_in_current_folder)
        logger.info(f'{get_current_function()}(): Processed {len(file_names_in_current_folder)} CSV '
                    f'files from folder: {folder}')
    data_array = np.array(data_list)
    logger.info(f'{get_current_function()}(): Processed a total of {len(data_list)} CSV '
                f'files and compiled into a {data_array.shape} data list.')
    return fldrs, all_file_names_list, data_array, perc_rect_list


def remove_top_n_rows_of_dataframe(in_df, n_rows: int = 1, copy=False):
    df = in_df.copy() if copy else in_df
    if n_rows < 0:
        err = f'Cannot remove negative rows from top of DataFrame. n_rows = {n_rows}'
        logger.error(err)
        raise ValueError(err)
    df = df[1:]  # Remove top n rows
    return df


@config.deco__log_entry_exit(logger)
def process_raw_data_and_filter_adaptively(df_input_data: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """ Legacy implementation.


    :param df_input_data: (DataFrame) expected: raw DataFrame of DLC results right after reading in using pandas.read_csv().
    EXAMPLE `df_input_data` input:
           scorer DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.1  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.2   ...
        1  coords                                              x                                                 y                                        likelihood   ...
        2       0                               1017.80322265625                                 673.5625610351562                                               1.0   ...
        3       1                             1018.4616088867188                                 663.2183837890625                                0.9999999403953552   ...
        4       2                             1018.5991821289062                                 663.4205322265625                                               1.0   ...
        5       3                             1013.0330810546875                                 651.7833251953125                                 0.999998927116394   ...

    :return
        : 2D array, filtered data
        : 1D array, percent filtered per BODYPART
    """
    # Type checking args
    if not isinstance(df_input_data, pd.DataFrame):
        raise TypeError(f'Input data was expected to be of type pandas.DataFrame but '
                        f'instead found: {type(df_input_data)}.')

    # Continue if args valid
    l_index, x_index, y_index, percent_filterd_per_bodypart__perc_rect = [], [], [], []

    # Remove top row. The top row only contained project name headers (e.g.: [scorer, DLCModel, DLCModel, DLCModel, ...]
    df_input_data_with_projectname_header_removed: pd.DataFrame = df_input_data[1:]

    # Convert data to raw array
    array_input_data_with_projectname_header_removed: np.ndarray = np.array(df_input_data_with_projectname_header_removed)

    # Loop over columns, aggregate which indices in the data fall under which category.
    #   x, y, and likelihood are the three main types of columns output from DLC.
    number_of_cols = len(array_input_data_with_projectname_header_removed[0])  # number_of_cols = len(array_input_data_with_top_row_removed[0])
    for header_idx in range(number_of_cols):  # range(len(currdf[0])):
        current_column_header = array_input_data_with_projectname_header_removed[0][header_idx]
        if current_column_header == "likelihood":
            l_index.append(header_idx)
        elif current_column_header == "x":
            x_index.append(header_idx)
        elif current_column_header == "y":
            y_index.append(header_idx)
        elif current_column_header == 'coords':
            pass  # Ignore. Usually this is the index column and is only seen once. No data in this column.
        else:  # Case: unexpected column suffice detected
            err = f'An inappropriate column header was found: ' \
                  f'{array_input_data_with_projectname_header_removed[0][header_idx]}.' \
                  f'Check on CSV to see if has an unexpected output format.'
            logger.error(err)
            raise ValueError(err)

    # Remove the first column (called "coords", the index which counts rows but has no useful data)
    array_input_data_without_coords = array_input_data_with_projectname_header_removed[:, 1:]

    # Slice data into separate arrays based on column names (derived earlier from the respective index)
    data_x = array_input_data_without_coords[:, np.array(x_index) - 1]
    data_y = array_input_data_without_coords[:, np.array(y_index) - 1]
    data_likelihood = array_input_data_without_coords[:, np.array(l_index) - 1]

    array_data_filtered = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))  # Initialized as zeroes with  # currdf_filt: np.ndarray = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))

    logger.debug(f'{get_current_function()}(): Computing data threshold to forward fill any sub-threshold (x,y)...')

    percent_filterd_per_bodypart__perc_rect = [0 for _ in range(data_likelihood.shape[1])]  # for _ in range(data_lh.shape[1]): perc_rect.append(0)

    # Loop over data and do adaptive filtering
    # logger.debug(f'{get_current_function()}(): Loop over data and do adaptive filtering.')

    for col_j in tqdm(range(data_likelihood.shape[1]), desc=f'Adaptively filtering data...'):
        # Get histogram. Number of bins defaults to 10.
        histogram, bin_edges = np.histogram(data_likelihood[1:, col_j].astype(np.float))

        rise_arr = np.where(np.diff(histogram) >= 0)

        # Sometimes np.where returns a tuple depending on input dims, but
        #   based on our usage here, it should be length of 1 anyways. Select first elem to get the array.
        if isinstance(rise_arr, tuple): rise_arr = rise_arr[0]

        # Get likelihood value based on value of first rise element.
        rise_0, rise_1 = rise_arr[0], rise_arr[1]
        if rise_0 > 1:
            likelihood: float = (bin_edges[rise_0] + bin_edges[rise_0-1]) / 2
        else:
            likelihood: float = (bin_edges[rise_1] + bin_edges[rise_1 - 1]) / 2

        # Strip off the labels at the top row
        data_likelihood_asfloat = data_likelihood[1:, col_j].astype(np.float)

        percent_filterd_per_bodypart__perc_rect[col_j] = np.sum(data_likelihood_asfloat < likelihood) / data_likelihood.shape[0]

        for row_i in range(1, data_likelihood.shape[0] - 1):
            if data_likelihood_asfloat[row_i] < likelihood:
                array_data_filtered[row_i, (2 * col_j):(2 * col_j + 2)] = array_data_filtered[row_i - 1, (2 * col_j):(2 * col_j + 2)]
            else:
                array_data_filtered[row_i, (2 * col_j):(2 * col_j + 2)] = np.hstack([data_x[row_i, col_j], data_y[row_i, col_j]])

    # Remove first row in data array (values are all zeroes)
    array_filtered_data_without_first_row = np.array(array_data_filtered[1:])

    # Convert all data to np.float
    final_array_filtered_data = array_filtered_data_without_first_row.astype(np.float)  # TODO: remove this line? np.float is just an alias for Python's built-in float

    return final_array_filtered_data, percent_filterd_per_bodypart__perc_rect


@config.deco__log_entry_exit(logger)
def adaptive_filter_data_app(input_df: pd.DataFrame, BODYPARTS: dict):  # TODO: rename function for clarity?
    """
    TODO: purpose
    :param input_df: object, csv data frame
    :param BODYPARTS:

    :return currdf_filt: 2D array, filtered data
    :return perc_rect: 1D array, percent filtered per BODYPART
    """
    l_index, x_index, y_index = [], [], []
    currdf = np.array(input_df[1:])
    for body_part_key in BODYPARTS:  # TODO: indexing off of BODYPARTS keys should cause an error?
        if currdf[0][body_part_key + 1] == "likelihood":
            l_index.append(body_part_key)
        elif currdf[0][body_part_key + 1] == "x":
            x_index.append(body_part_key)
        elif currdf[0][body_part_key + 1] == "y":
            y_index.append(body_part_key)

    logger.debug('Extracting likelihood value...')
    curr_df1 = currdf[:, 1:]
    data_x = curr_df1[1:, np.array(x_index)]
    data_y = curr_df1[1:, np.array(y_index)]
    data_lh = curr_df1[1:, np.array(l_index)]
    currdf_filt = np.zeros((data_x.shape[0] - 1, (data_x.shape[1]) * 2))
    perc_rect = []

    logger.debug('Computing data threshold to forward fill any sub-threshold (x,y)...')
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[1:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([data_x[0, x], data_y[0, x]])
        for i in range(1, data_lh.shape[0] - 1):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([data_x[i, x], data_y[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect


########################################################################################################################
### Legacy functions

def main(folders: List[str]) -> None:
    """
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered data list
    :retrun perc_rect: 1D array, percent filtered per BODYPART
    """
    replacement_func = import_csvs_data_from_folders_in_PROJECTPATH_and_process_data
    err = f'This function, bsoid.util.likelihoodprocessing.main(), will be '\
          f'deprecated in future. Directly use {replacement_func.__qualname__} instead. '\
          f'Caller = {inspect.stack()[1][3]}'
    logger.error(err)
    raise Exception(err)
    # filenames, data, perc_rect = replacement_func(folders)
    # return filenames, data, perc_rect
def get_filenames(folder: str):
    """
    Gets a list of CSV filenames within a folder (assuming it exists within BASE_PATH)
    :param folder: str, folder path
    :return: list, filenames
    """
    replacement_func = get_filenames_csvs_from_folders_recursively_in_dlc_project_path
    logger.warn(f'**NOTE: this function implicitly assume the argument folder resides in BASE_PATH***. '
                f'`folder` argument value = {folder} . Replacement function is '
                f'currently: {get_filenames_csvs_from_folders_recursively_in_dlc_project_path.__qualname__}. '
                f'This function is likely to be deprecated in the future. Caller = {inspect.stack()[1][3]}')
    return replacement_func(folder)


def adaptive_filter_LEGACY(df_input_data: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """
    Deprecation warning. Do not alter this function so that we can confirm new function output matches old function.
    """
    logger.warn(f'{inspect.stack()[0][3]}(): will be deprecated in future. '
                f'Instead, try using: {process_raw_data_and_filter_adaptively.__qualname__}')
    # Type checking args
    if not isinstance(df_input_data, pd.DataFrame):
        raise TypeError(f'`df_input` was expected to be of type pandas.DataFrame but '
                        f'instead found: {type(df_input_data)}.')
    # Continue args valid
    l_index, x_index, y_index, percent_filterd_per_bodypart__perc_rect = [], [], [], []
    # Remove top row. It contains COLUMN LABELS
    df_input_data_with_top_row_removed: pd.DataFrame = df_input_data[1:]
    array_input_data_with_top_row_removed: np.ndarray = np.array(df_input_data_with_top_row_removed)
    # currdf = array_input_data_with_top_row_removed

    # Loop over columns, aggregate which indices in the data fall under which category.
    #   x, y, and likelihood are the three main types of columns output from DLC.
    number_of_cols = len(array_input_data_with_top_row_removed[0])
    for header_idx in range(number_of_cols):  # range(len(currdf[0])):
        current_column_header = array_input_data_with_top_row_removed[0][header_idx]
        if current_column_header == "likelihood":
            l_index.append(header_idx)
        elif current_column_header == "x":
            x_index.append(header_idx)
        elif current_column_header == "y":
            y_index.append(header_idx)
        elif current_column_header == 'coords':
            pass  # Ignore. Usually this is the title of the index column and is only seen once. No data to be had here.
        else:
            err = f'An inappropriate column header was found: {array_input_data_with_top_row_removed[0][header_idx]}'  # TODO: elaborate on error
            logger.error(err)
            raise ValueError(err)

    logger.debug(f'{get_current_function()}: Extracting likelihood value...')
    curr_df1 = array_input_data_with_top_row_removed[:, 1:]
    data_x = curr_df1[:, np.array(x_index) - 1]
    data_y = curr_df1[:, np.array(y_index) - 1]
    data_lh = curr_df1[:, np.array(l_index) - 1]
    currdf_filt: np.ndarray = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))  # Initialized as zeroes with  # currdf_filt: np.ndarray = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))

    logger.info('Computing data threshold to forward fill any sub-threshold (x,y)...')
    percent_filterd_per_bodypart__perc_rect = [0 for _ in range(data_lh.shape[1])]  # for _ in range(data_lh.shape[1]): perc_rect.append(0)

    for x in tqdm(range(data_lh.shape[1])):
        histogram, bin_edges = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(histogram) >= 0)
        if rise_a[0][0] > 1:
            llh = ((bin_edges[rise_a[0][0]] + bin_edges[rise_a[0][0]-1]) / 2)
        else:
            llh = ((bin_edges[rise_a[0][1]] + bin_edges[rise_a[0][1]-1]) / 2)
        data_lh_float = data_lh[1:, x].astype(np.float)
        percent_filterd_per_bodypart__perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        for i in range(1, data_lh.shape[0] - 1):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([data_x[i, x], data_y[i, x]])
    currdf_filt = np.array(currdf_filt[1:])
    currdf_filt = currdf_filt.astype(np.float)

    return currdf_filt, percent_filterd_per_bodypart__perc_rect


########################################################################################################################

if __name__ == '__main__':
    # shared_test_file =
    # df_current_file = pd.read_csv(filename, low_memory=False)
    # curr_df_filt, perc_rect = adaptive_filter_data(df_current_file)
    # x = import_csvs_data_from_folders_in_PROJECTPATH_and_process_data

    pass
