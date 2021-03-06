"""
likelihood processing utilities
Forward fill low likelihood (x,y)
"""

from bsoid_py.config.LOCAL_CONFIG import BASE_PATH, FRAME_DIR, SHORTVID_DIR, TRAIN_FOLDERS
# from bsoid_py.utils.likelihoodprocessing import sort_nicely

from typing import List, Tuple
from tqdm import tqdm
import glob
import logging
import pandas as pd
import numpy as np
import os
import pandas as pd
import re
import warnings


def boxcar_center(a, n):
    """
    TODO
    :param a: TODO
    :param n: TODO
    :return: TODO
    """
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg


def convert_int(s):
    """ Converts digit string to integer """
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        e.g.: "z23a" -> ["z", 23, "a"]
    """
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def get_filenames(folder):
    """
    Gets a list of CSV filenames within a folder (assuming it exists within BASE_PATH)
    :param folder: str, folder path
    :return: list, filenames
    """
    replacement_func = get_filenames_csvs_from_folders_recursively_in_basepath
    warnings.warn('**NOTE: this function implicitly assume the argument folder resides in BASE_PATH***. '
                  f'`folder` argument value = {folder} . Replacement function is '
                  f'currently: {get_filenames_csvs_from_folders_recursively_in_basepath.__qualname__}. '
                  f'This function is likely to be deprecated in the future.')

    return replacement_func(folder)


def get_filenames_csvs_from_folders_recursively_in_basepath(folder: str):
    """
    Get_filenames() makes the assumption that the folder is in BASEPATH; however, it is an obfuscated assumption
    and bad. A new function that DOES NOT RESOLVE PATH IMPLICITLY WITHIN should be created and used.
    :param folder:
    :return:
    """
    path_to_check_for_csvs = BASE_PATH + folder + os.path.sep+'**'+os.path.sep+'*.csv'
    logging.debug(f'Path that is being checked with "glob": {path_to_check_for_csvs}')
    filenames = glob.glob(path_to_check_for_csvs, recursive=True)
    sort_nicely(filenames)
    logging.info(f'files found: {filenames}')
    return filenames


def import_csvs_data_from_folders_in_BASEPATH(folders: List[str]) -> Tuple[List, np.ndarray, List]:
    """
    Import multiple folders containing .csv files and process them
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    # TODO: what does `raw_data_list` do? It looks like a variable without a purpose. It appends but does not return.
    file_names_list, data_list, perc_rect_li = [], [], []
    for idx_folder, folder in enumerate(folders):  # Loop through folders
        filenames_found_in_current_folder = get_filenames_csvs_from_folders_recursively_in_basepath(folder)
        for idx_filename, filename in enumerate(filenames_found_in_current_folder):
            logging.info(f'Importing CSV file {idx_filename+1} from folder {idx_folder+1}')
            # Read CSV in as DataFrame
            df_current_file = pd.read_csv(filename, low_memory=False)
            # Modify read-in DataFrame
            df_current_file_filtered, perc_rect = adaptive_filter(df_current_file)
            logging.info(f'Done preprocessing (x,y) from file {idx_filename+1}, folder {idx_folder+1}.')
            # # raw_data_list.append(df_current_file)
            # Append perc_rect and DataFrame to respective lists
            perc_rect_li.append(perc_rect)
            data_list.append(df_current_file_filtered)
        file_names_list.append(filenames_found_in_current_folder)
        logging.info(f'Processed {len(filenames_found_in_current_folder)} CSV files from folder: {folder}')
    data_array: np.ndarray = np.array(data_list)
    logging.info(f'Processed a total of {len(data_list)} CSV files, and compiled into a {data_array.shape} data list.')
    return file_names_list, data_array, perc_rect_li


def adaptive_filter(df_input: pd.DataFrame):
    """
    :param df_input: object, csv data frame
    :return currdf_filt: 2D array, filtered data
    :return perc_rect: 1D array, percent filtered per BODYPART
    """
    l_index = []
    x_index = []
    y_index = []
    df_input = np.array(df_input[1:])
    for header_idx in range(len(df_input[0])):
        if df_input[0][header_idx] == "likelihood":
            l_index.append(header_idx)
        elif df_input[0][header_idx] == "x":
            x_index.append(header_idx)
        elif df_input[0][header_idx] == "y":
            y_index.append(header_idx)
    logging.info('Extracting likelihood value...')
    curr_df1 = df_input[:, 1:]
    data_x = curr_df1[:, np.array(x_index) - 1]
    data_y = curr_df1[:, np.array(y_index) - 1]
    data_lh = curr_df1[:, np.array(l_index) - 1]
    currdf_filt = np.zeros((data_x.shape[0] - 1, (data_x.shape[1]) * 2))
    perc_rect = []
    logging.info('Computing data threshold to forward fill any sub-threshold (x,y)...')

    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = ((b[rise_a[0][0]] + b[rise_a[0][0]-1]) / 2)
        else:
            llh = ((b[rise_a[0][1]] + b[rise_a[0][1]-1]) / 2)
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


def main(folders: List[str]):
    """
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered data list
    :retrun perc_rect: 1D array, percent filtered per BODYPART
    """
    filenames, data, perc_rect = import_csvs_data_from_folders_in_BASEPATH(folders)
    return filenames, data, perc_rect


if __name__ == '__main__':
    main(TRAIN_FOLDERS)
