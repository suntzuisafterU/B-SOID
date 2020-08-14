"""
likelihood processing utilities
Forward fill low likelihood (x,y)
"""

from typing import List, Tuple
from tqdm import tqdm
import glob
import logging
import numpy as np
import pandas as pd
import re
import warnings

from bsoid.config.LOCAL_CONFIG import BASE_PATH, TRAIN_FOLDERS


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


def alphanum_key(s) -> List:
    """ Turn a string into a list of string and number chunks.
        e.g.: input: "z23a" -> output: ["z", 23, "a"]
    """
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l: list):
    """ Sort the given list in the way that humans expect. """
    if not isinstance(l, list):
        raise TypeError(f'argument `l` expected to be of type list but instead found: {type(l)} (value: {l})')
    l.sort(key=alphanum_key)


def get_filenames(folder):
    """
    Gets a list of filenames within a folder (assuming it exists within BASE_PATH)
    :param folder: str, folder path
    :return: list, filenames
    """
    filenames = glob.glob(BASE_PATH + folder + '/*.csv')
    sort_nicely(filenames)
    return filenames


def import_folders(folders: list) -> Tuple[List, np.ndarray, List]:
    """
    Import multiple folders containing .csv files and process them
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    filenames = []
    rawdata_li = []
    data_li = []
    perc_rect_li = []
    for i, fd in enumerate(folders):  # Loop through folders
        f = get_filenames(fd)
        for j, filename in enumerate(f):
            logging.info(f'Importing CSV file {j + 1} from folder {i + 1}')
            curr_df = pd.read_csv(filename, low_memory=False)
            curr_df_filt, perc_rect = adp_filt(curr_df)
            logging.info('Done preprocessing (x,y) from file {}, folder {}.'.format(j + 1, i + 1))
            rawdata_li.append(curr_df)
            perc_rect_li.append(perc_rect)
            data_li.append(curr_df_filt)
        filenames.append(f)
        logging.info('Processed {} CSV files from folder: {}'.format(len(f), fd))
    data = np.array(data_li)
    logging.info('Processed a total of {} CSV files, and compiled into a {} data list.'.format(len(data_li),
                                                                                               data.shape))
    return filenames, data, perc_rect_li


def adp_filt(currdf: object):  # TODO: low: rename for clarity
    """
    TODO: purpose
    :param currdf: object, csv data frame
    :return currdf_filt: 2D array, filtered data
    :return perc_rect: 1D array, percent filtered per BODYPART
    """
    l_index = []
    x_index = []
    y_index = []
    currdf = np.array(currdf[1:])
    for header in range(len(currdf[0])):
        if currdf[0][header] == "likelihood":
            l_index.append(header)
        elif currdf[0][header] == "x":
            x_index.append(header)
        elif currdf[0][header] == "y":
            y_index.append(header)
    logging.info('Extracting likelihood value...')
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[:, np.array(x_index) - 1]
    datay = curr_df1[:, np.array(y_index) - 1]
    data_lh = curr_df1[:, np.array(l_index) - 1]
    currdf_filt = np.zeros((datax.shape[0] - 1, (datax.shape[1]) * 2))
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
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
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
    warnings.warn('This function, main(), will be deprecated in future as it adds nothing except obfuscation.')
    filenames, data, perc_rect = import_folders(folders)
    return filenames, data, perc_rect


if __name__ == '__main__':
    # TODO: should this module even have a main if it's just a helper module?
    import_folders(TRAIN_FOLDERS)
