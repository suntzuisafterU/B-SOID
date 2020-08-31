"""
likelihood processing utilities
Forward fill low likelihood (x,y)
"""

from typing import Any, List, Tuple
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


def convert_int(s: str):
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
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_list_nicely_in_place(list_input: list) -> None:
    """ Sort the given list (in place) in the way that humans expect. """
    if not isinstance(list_input, list):
        raise TypeError(f'argument `l` expected to be of type list but '
                        f'instead found: {type(list_input)} (value: {list_input}).')
    list_input.sort(key=alphanum_key)


def get_filenames(folder):
    """
    Gets a list of CSV filenames within a folder (assuming it exists within BASE_PATH)
    :param folder: str, folder path
    :return: list, filenames
    """
    replacement_func = get_filenames_csvs_from_folders_recursively_in_basepath
    logger.warn('**NOTE: this function implicitly assume the argument folder resides in BASE_PATH***. '
                  f'`folder` argument value = {folder} . Replacement function is '
                  f'currently: {get_filenames_csvs_from_folders_recursively_in_basepath.__qualname__}. '
                  f'This function is likely to be deprecated in the future. Caller = {inspect.stack()[1][3]}')

    return replacement_func(folder)


def get_filenames_csvs_from_folders_recursively_in_basepath(folder: str) -> List:
    """
    Get_filenames() makes the assumption that the folder is in BASEPATH; however, it is an obfuscated assumption
    and bad. A new function that DOES NOT RESOLVE PATH IMPLICITLY WITHIN should be created and used.
    :param folder:
    :return:
    """
    path_to_check_for_csvs = f'{config.DLC_PROJECT_PATH}{os.path.sep}{folder}{os.path.sep}**{os.path.sep}*.csv'
    logger.debug(f'Path that is being checked with using glob selection: {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=True)
    sort_list_nicely_in_place(filenames)
    logger.info(f'List of files found: {filenames}. Total files found : {len(filenames)}.')
    return filenames


def import_csvs_data_from_folders_in_BASEPATH_and_process_data(folders: list) -> Tuple[List, np.ndarray, List]:
    """
    Import multiple folders containing .csv files and process them
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    # TODO: what does `raw_data_list` do? It looks like a variable without a purpose. It appends but does not return.
    file_names_list, raw_data_list, data_list, perc_rect_list = [], [], [], []
    if len(folders) == 0:
        raise ValueError(f'submitted folders list is empty')
        # pass
    for idx_folder, folder in enumerate(folders):  # Loop through folders
        filenames_found_in_current_folder = get_filenames_csvs_from_folders_recursively_in_basepath(folder)
        for idx_filename, filename in enumerate(filenames_found_in_current_folder):
            logger.info(f'Importing CSV file {idx_filename+1} from folder {idx_folder+1}')
            df_current_file = pd.read_csv(filename, low_memory=False)
            curr_df_filt, perc_rect = adaptive_filter_data(df_current_file)
            logger.info(f'Done preprocessing (x,y) from file #{idx_filename+1}, folder #{idx_folder+1}.')
            raw_data_list.append(df_current_file)
            perc_rect_list.append(perc_rect)
            data_list.append(curr_df_filt)
        file_names_list.append(filenames_found_in_current_folder)
        logger.info(f'Processed {len(filenames_found_in_current_folder)} CSV files from folder: {folder}')
    data_array: np.ndarray = np.array(data_list)
    logger.info(f'Processed a total of {len(data_list)} CSV files and compiled into a {data_array.shape} data list.')
    return file_names_list, data_array, perc_rect_list


def adaptive_filter_data(df_input: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """
    TODO: low: purpose
    :param df_input: (DataFrame) TODO
    :param currdf: object, csv data frame
    :return currdf_filt: 2D array, filtered data
    :return perc_rect: 1D array, percent filtered per BODYPART
    """
    # Type checking args
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError(f'`df_input` was expected to be of type pandas.DataFrame but '
                        f'instead found: {type(df_input)}.')
    # Continue if valid
    l_index, x_index, y_index, perc_rect = [], [], [], []
    currdf = np.array(df_input[1:])
    for header_idx in range(len(currdf[0])):
        if currdf[0][header_idx] == "likelihood":
            l_index.append(header_idx)
        elif currdf[0][header_idx] == "x":
            x_index.append(header_idx)
        elif currdf[0][header_idx] == "y":
            y_index.append(header_idx)
        else: pass  # TODO: low: should this be failing silently?

    logger.info('Extracting likelihood value...')
    curr_df1 = currdf[:, 1:]
    data_x = curr_df1[:, np.array(x_index) - 1]
    data_y = curr_df1[:, np.array(y_index) - 1]
    data_lh = curr_df1[:, np.array(l_index) - 1]
    currdf_filt: np.ndarray = np.zeros((data_x.shape[0] - 1, (data_x.shape[1]) * 2))

    logger.info('Computing data threshold to forward fill any sub-threshold (x,y)...')
    for _ in range(data_lh.shape[1]):  # TODO: low: simplify `perc_rect` initialization as list of zeroes
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        histogram, bin_edges = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(histogram) >= 0)
        if rise_a[0][0] > 1:
            llh = ((bin_edges[rise_a[0][0]] + bin_edges[rise_a[0][0]-1]) / 2)
        else:
            llh = ((bin_edges[rise_a[0][1]] + bin_edges[rise_a[0][1]-1]) / 2)
        data_lh_float = data_lh[1:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        for i in range(1, data_lh.shape[0] - 1):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([data_x[i, x], data_y[i, x]])
    currdf_filt = np.array(currdf_filt[1:])
    currdf_filt = currdf_filt.astype(np.float)

    return currdf_filt, perc_rect


# Legacy functions. Will be potentially deleted later.
def main(folders: List[str]) -> None:
    """
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered data list
    :retrun perc_rect: 1D array, percent filtered per BODYPART
    """
    replacement_func = import_csvs_data_from_folders_in_BASEPATH_and_process_data
    err = f'This function, bsoid.util.likelihoodprocessing.main(), will be '\
          f'deprecated in future. Directly use {replacement_func.__qualname__} instead. '\
          f'Caller = {inspect.stack()[1][3]}'
    logger.error(err)
    # raise NotImplementedError(err)
    # logger.warn()
    # filenames, data, perc_rect = replacement_func(folders)
    # return filenames, data, perc_rect


########################################################################################################################

if __name__ == '__main__':
    # TODO: should this module even have a main if it's just a helper module?
    # import_csvs_data_from_folders_in_BASEPATH(config.TRAIN_FOLDERS)
    pass
