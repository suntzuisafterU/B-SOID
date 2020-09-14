"""
Summary statistics

    This bsoid module version of statistics.py aggregates all previous submodules which
did not adhere to DRY standards and creates a single source of statistics utility.
"""

from typing import List, Tuple
import inspect
import numpy as np
import pandas as pd

from bsoid import config


logger = config.initialize_logger(__name__)


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


def rle(in_array) -> Tuple:  # TODO: rename function for clarity?
    """
    Run length encoding. Partial credit to R rle function. Multi datatype arrays catered-for including non-Numpy.

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


def behv_time(labels: np.ndarray):  # TODO: rename function for clarity?
    """
    TODO: med: purpose
    :param labels: 1D array, predicted labels
    :return beh_t: 1D array, percent time for each label
    """
    # Ensure argument types
    if not isinstance(labels, np.ndarray):
        raise TypeError(f'Argument `labels` was expected to be of type np.ndarray but '
                        f'instead found {type(labels)} (value: {labels}).')
    # TODO: rename variables for clarity?
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


# TODO: rename main()? Should only be called "main" if this module is called at runtime as standalone file?
def main(labels) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (Original function implementation across _py, _umap, and _voc submodules. Kept, for now, for
        backwards compatibility reasons.)
    """
    replacement_func = get_runlengths_statistics_transition_matrix_from_labels
    logger.error(f'This function, bsoid.util.statistics.main(), will be deprecated in future. Check back for a '
                 f'renamed/refactored version later. '
                 f'Current replacement: {get_runlengths_statistics_transition_matrix_from_labels.__qualname__}.'
                 f'CALLER = {inspect.stack()[0][3]}.')
    return replacement_func(labels)

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


### LEGACY FUNCTIONS ###################################################################################################

def feat_dist(features: np.ndarray) -> Tuple[List, List, List, List]:
    """   *** DEPRECATION WARNING ***   """
    replacement_func = get_feature_distribution
    err = f'This function, feat_dist, will be deprecated in future. Instead, replace its use with ' \
          f'{replacement_func.__qualname__} (potential caller of this function = {inspect.stack()[1][3]}).'
    logger.warning(err)
    return replacement_func(features)
def main_app(labels, n):
    """
    TODO: why is the _app version different?
    Pre-existing "main()" function from the bsoid_app legacy submodule.
    :param labels: 1D array: predicted labels
    :param output_path: string, output directory
    :return dur_stats: object, behavioral duration statistics data frame
    :return tm: object, transition matrix data frame
    """
    runlen_df, dur_stats = behv_dur(labels)
    B, df_tm, B_norm = transition_matrix_app(labels, n)
    return runlen_df, dur_stats, B, df_tm, B_norm