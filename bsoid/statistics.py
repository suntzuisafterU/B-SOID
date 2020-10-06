"""

"""

from typing import Any, Dict, List, Tuple, Union
import functools
import inspect
import numpy as np
import pandas as pd

from bsoid import config


logger = config.initialize_logger(__name__)



# LLHPROC

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


### Legacy functions ###################################################################################################

def import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(folders_in_project_path: list) -> Tuple[List[List[str]], List[np.ndarray], List]:
    """
    Import multiple folders containing .csv files and process them
    :param folders_in_project_path: List[str]: Data folders
    :return filenames: list, data filenames
    :return data: List of arrays, filtered csv data
    :return perc_rect_li: list, percent filtered
    """
    # TODO: what does `raw_data_list` do? It looks like a variable without a purpose. It appends but does not return.
    warning = f''
    logger.warning(warning)
    return io.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(folders_in_project_path)


def main(folders: List[str]) -> None:
    """
    :param folders: list, data folders
    :return filenames: list, data filenames
    :return data: list, filtered data list
    :retrun perc_rect: 1D array, percent filtered per BODYPART
    """
    replacement_func = import_csvs_data_from_folders_in_PROJECTPATH_and_process_data
    err = f'This function, {inspect.stack()[0][3]}(), will be '\
          f'deprecated in future. Directly use {replacement_func.__qualname__} instead. '\
          f'Caller = {inspect.stack()[1][3]}'
    logger.error(err)
    raise Exception(err)
    # filenames, data, perc_rect = replacement_func(folders)
    # return filenames, data, perc_rect


def adaptive_filter_LEGACY(df_input_data: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """
    Deprecation warning. Do not alter this function so that we can confirm new function output matches old function.
    """
    logger.warning(f'{inspect.stack()[0][3]}(): will be deprecated in future. '
                   f'Instead, try using: {feature_engineering.process_raw_data_and_filter_adaptively.__qualname__}')
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

# Stats
def mean(*args):
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    return functools.reduce(lambda x, y: x + y, args, 0) / len(args)


def sum_args(*args):
    args = [arg for arg in args if arg == arg]  # Remove any 'nan' values
    return functools.reduce(lambda x, y: x + y, args, 0)


def get_feature_distribution(features: np.ndarray):
    """
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


def transition_matrix_app(labels, n: int) -> Tuple:
    """
    TODO: purpose
    :param n: TODO
    :param labels: 1D array, predicted labels
    :return df_tm: object, transition matrix data frame
    """
    # n = 1 + max(labels)
    tm = [[0] * n for _ in range(n)]
    for (i, j) in zip(labels, labels[1:]):
        tm[i][j] += 1
    B = np.matrix(tm)  # TODO: HIGH: numpy error: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    df_tm = pd.DataFrame(tm)
    B = np.matrix(tm)
    B_norm = B / B.sum(axis=1)
    return B, df_tm, B_norm


# TODO: HIGH: reconcile below transition_matrix() and above
def transition_matrix(labels) -> pd.DataFrame:  # source: bsoid_py, bsoid_umap, bsoid_voc
    """
    TODO: purpose
    :param labels: 1D array, predicted labels
    :return df_transition_matrix: (DataFrame) Transition matrix DataFrame
    """
    n = 1 + max(labels)
    transition_matrix = [[0] * n for _ in range(n)]
    for i, j in zip(labels, labels[1:]):
        transition_matrix[i][j] += 1
    for row in transition_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    df_transition_matrix = pd.DataFrame(transition_matrix)
    return df_transition_matrix


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


@config.deco__log_entry_exit(logger)
def get_runlengths_statistics_transition_matrix_from_labels(labels) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ TODO: rename function for concision when purpose made clearer
    TODO: med: purpose
    :param labels: 1D array: predicted labels
    :returns
        df_runlengths: (DataFrame)  TODO
        df_dur_statistics: (DataFrame) behavioral duration statistics data frame
        tm: (DataFrame) transition matrix data frame
    """
    df_runlengths, df_dur_statistics = behv_dur(labels)
    tm = transition_matrix(labels)
    return df_runlengths, df_dur_statistics, tm