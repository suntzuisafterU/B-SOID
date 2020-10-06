"""

"""
from typing import Any, Dict, List, Tuple, Union
import functools
import inspect
import numpy as np
import pandas as pd
import re

from bsoid import config


logger = config.initialize_logger(__name__)

# LLHPROC


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


@config.deco__log_entry_exit(logger)
def augmented_runlength_encoding(labels: Union[List, np.ndarray]) -> Tuple[List[Any], List[int], List[int]]:
    """
    TODO: med: purpose // purpose unclear
    :param labels: (list or np.ndarray) predicted labels
    :return
        label_list: (list) the label number
        idx: (list) label start index
        lengths: (list) how long each bout lasted for
    """
    label_list, idx_list, lengths_list = [], [], []
    i = 0
    while i < len(labels) - 1:
        # 1/3: Record current index
        idx_list.append(i)
        # 2/3: Record current label
        current_label = labels[i]
        label_list.append(current_label)
        # Iterate over i while current label and next label are same
        start_index = i
        while i < len(labels)-1 and labels[i] == labels[i + 1]:
            i += 1
        end_index = i
        # 3/3: Calculate length of repetitions, then record lengths to list
        length = end_index - start_index
        lengths_list.append(length)
        # Increment and continue
        i += 1
    return label_list, idx_list, lengths_list


# Stats
def mean(*args):
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    return functools.reduce(lambda x, y: x + y, args, 0) / len(args)


def sum_args(*args):
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    return functools.reduce(lambda x, y: x + y, args, 0)


def get_feature_distribution(features: np.ndarray):
    """ TODO: investigate this funtion from legacy
    TODO: low: purpose
    :param features: (ndarray) TODO
    :return: Tuple TODO
        feature_range:
        feature_median:
        p_cts:
        edges:
    """
    if not isinstance(features, np.ndarray):
        raise TypeError(f"Argument `features` expected to be of type np.ndarray but instead "
                        f"found {type(features)} (value: {features}).")
    feature_range, feature_median, p_cts, edges = [], [], [], []
    # Iterates over rows so that ___
    for i in range(features.shape[0]):
        feature_range.append(
            [np.quantile(features[i, :], 0.05),
             np.quantile(features[i, :], 0.95), ]
        )
        feature_median.append(np.quantile(features[i, :], 0.5))
        p_ct, edge = np.histogram(features[i, :], 50, density=True)
        p_cts.append(p_ct)
        edges.append(edge)
    return feature_range, feature_median, p_cts, edges


def rle(in_array) -> Union[Tuple[None, None, None], Tuple[Any, Any, Any]]:
    """
    Run length encoding. Partial credit to R's rle() function. Multi datatype arrays catered-for including non-Numpy.

    {
        - R Documentation excerpt -
        Run Length Encoding (rle)
        Description:
        Compute the lengths and values of runs of equal values in a vector – or the reverse operation.

            EXAMPLE (inputs to console denoted by '>'):
        > x <- rev(rep(6:10, 1:5))
        > x
        [1] 10 10 10 10 10  9  9  9  9  8  8  8  7  7  6
        > rle(x)
        Run Length Encoding
          lengths: int [1:5] 5 4 3 2 1
          values : int [1:5] 10 9 8 7 6
        (Author's note: 'values' at bottom occur 'lengths' times specified above each value
    }

    :param in_array: (ndarray) TODO
    :returns:
        run_lengths: (list) TODO
        start_positions: (list) TODO
        values: (list) TODO
        """

    array = np.asarray(in_array)  # Force into numpy array type
    num_array_elements = len(array)
    if num_array_elements != 0:
        y = np.array(array[1:] != array[:-1])                           # Pairwise unequal (string safe)
        i = np.append(np.where(y), num_array_elements - 1)              # Must include last element position
        run_lengths = np.diff(np.append(-1, i))                         # Run lengths
        start_positions = np.cumsum(np.append(0, run_lengths))[:-1]     # Positions
        values = array[i]
        return run_lengths, start_positions, values
    return None, None, None


def behv_time(labels: np.ndarray) -> List:  # TODO: rename function for clarity
    """
    TODO: med: purpose
    :param labels: 1D array, predicted labels
    :return beh_t: 1D array, percent time for each label
    """
    # Ensure argument types
    if not isinstance(labels, np.ndarray):
        raise TypeError(f'Argument `labels` was expected to be of type np.ndarray but '
                        f'instead found {type(labels)} (value: {labels}).')
    # TODO: rename variables for clarity
    beh_t = []
    for i in range(len(np.unique(labels))):
        t = np.sum(labels == i) / labels.shape[0]
        beh_t.append(t)
    return beh_t


def behv_dur(labels) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    # TODO: med: purpose
    :param labels: 1D array, predicted labels
    :return runlen_df: object, behavioral duration run lengths data frame
    :return dur_stats: object, behavioral duration statistics data frame
    """
    # Create runlengths DataFrame
    run_lengths, start_positions, values = rle(labels)
    df_lengths = pd.DataFrame(run_lengths, columns={'Run lengths'})
    df_grp = pd.DataFrame(values, columns={'B-SOiD labels'})
    df_positions = pd.DataFrame(start_positions, columns={'Start time (frames)'})
    df_runlengths = pd.concat([df_grp, df_positions, df_lengths], axis=1)

    # Create duration statistics DataFrame
    beh_t = behv_time(labels)
    dur_means, dur_quantile10, dur_quantile25, dur_quantile50, dur_quantile75, dur_quantile90 = [], [], [], [], [], []
    for i in range(len(np.unique(values))):
        try:
            dur_means.append(np.mean(run_lengths[np.where(values == i)]))
            dur_quantile10.append(np.quantile(run_lengths[np.where(values == i)], 0.10))
            dur_quantile25.append(np.quantile(run_lengths[np.where(values == i)], 0.25))
            dur_quantile50.append(np.quantile(run_lengths[np.where(values == i)], 0.50))
            dur_quantile75.append(np.quantile(run_lengths[np.where(values == i)], 0.75))
            dur_quantile90.append(np.quantile(run_lengths[np.where(values == i)], 0.90))
        except:  # TODO: med: exception too broad. If it fails mid-way thru, unequal final list lengths can result. I have a feeling that the error is found at dur_quantile10 on each loop.
            # dur_means.append(0)
            dur_quantile10.append(0)
            dur_quantile25.append(0)
            dur_quantile50.append(0)
            dur_quantile75.append(0)
            dur_quantile90.append(0)

    all_duration_data_as_array = np.concatenate([
        np.array(beh_t).reshape(len(np.array(beh_t)), 1),
        np.array(dur_means).reshape(len(np.array(dur_means)), 1),
        np.array(dur_quantile10).reshape(len(np.array(dur_quantile10)), 1),
        np.array(dur_quantile25).reshape(len(np.array(dur_quantile25)), 1),
        np.array(dur_quantile50).reshape(len(np.array(dur_quantile50)), 1),
        np.array(dur_quantile75).reshape(len(np.array(dur_quantile75)), 1),
        np.array(dur_quantile90).reshape(len(np.array(dur_quantile90)), 1)
    ], axis=1)
    duration_statistics_columns = pd.MultiIndex.from_tuples([('Stats',  'Percent of time'),
                                                             ('',       'Mean duration (frames)'),
                                                             ('',       '10th %tile (frames)'),
                                                             ('',       '25th %tile (frames)'),
                                                             ('',       '50th %tile (frames)'),
                                                             ('',       '75th %tile (frames)'),
                                                             ('',       '90th %tile (frames)')],
                                                            names=['', 'B-SOiD labels'])
    df_dur_statistics = pd.DataFrame(all_duration_data_as_array, columns=duration_statistics_columns)

    return df_runlengths, df_dur_statistics
