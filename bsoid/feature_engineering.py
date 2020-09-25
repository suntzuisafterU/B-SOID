"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.

Potential abbreviations:
    sn: snout
    pt: proximal tail ?

DELETE THIS STRING
7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
    1. body length (or "[d_ST]"): distance from snout to base of tail
    2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT],
        where [d_FT] is the distance between front paws and base of tail
    3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
    4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

    5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
    6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
    7. snout to base-of-tail change in angle:

Author also specifies that: the features are also smoothed over, or averaged across,
    a sliding window of size equivalent to 60ms (30ms prior to and after the frame of interest).
"""
from bhtsne import tsne as TSNE_bhtsne
from sklearn import mixture
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import hdbscan
import inspect
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import umap

from bsoid import config
from bsoid.util import check_arg, likelihoodprocessing, visuals

logger = config.initialize_logger(__name__)


########################################################################################################################

def adaptively_filter_dlc_output(in_df: pd.DataFrame, copy=False) -> Tuple[pd.DataFrame, List[float]]:  # TODO: implement new adaptive-filter_data for new data pipelineing
    """ *NEW* --> Successor function to old method in likelikhood processing. Uses new DF format.
    Takes in a ____ TODO...

    (Described as adaptive high-pass filter by original author)
    Follows same form as legacy only for continuity reasons. Can be refactored for performance later.

    Note: you'll end up with 1 less row as a result than you started with.
    :param in_df:
    :param df_input_data: (DataFrame) expected: raw DataFrame of DLC results right after
        reading in using bsoid.read_csv().

        EXAMPLE `df_input_data` input:  # TODO: remove bodyparts_coords col? Check bsoid.io.read_csv() return format.
              bodyparts_coords        Snout/Head_x       Snout/Head_y Snout/Head_likelihood Forepaw/Shoulder1_x Forepaw/Shoulder1_y Forepaw/Shoulder1_likelihood  ...                                          scorer
            0                0     1013.7373046875   661.953857421875                   1.0  1020.1138305664062   621.7146606445312           0.9999985694885254  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
            1                1  1012.7627563476562  660.2426147460938                   1.0  1020.0912475585938   622.9310913085938           0.9999995231628418  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
            2                2  1012.5982666015625   660.308349609375                   1.0  1020.1837768554688   623.5087280273438           0.9999994039535522  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
            3                3  1013.2752685546875  661.3504028320312                   1.0     1020.6982421875   624.2875366210938           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
            4                4  1013.4093017578125  661.3643188476562                   1.0  1020.6074829101562     624.48486328125           0.9999998807907104  ...  DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000
    :param copy: (bool) Indicates whether to create an entirely new DataFrame object as a result so that
        the original input DataFrame is not changed afterwards.

    :return
        : DataFrame of filtered data
        : 1D array, percent filtered per BODYPART
    """
    # Checking args
    if not isinstance(in_df, pd.DataFrame):
        type_err = f'Input data was expected to be of type pandas.DataFrame but instead found: {type(in_df)}.'
        logger.error(type_err)
        raise TypeError(type_err)

    # Continue
    if 'scorer' not in in_df.columns:
        col_not_found_err = f'TODO: "scorer" col not found but should exist // All columns: {in_df.columns}'
        logger.error(col_not_found_err)
        raise ValueError(col_not_found_err)

    scorer_values = np.unique(in_df['scorer'])
    if len(scorer_values) != 1:
        err = f'TODO: there should be 1 unique scorer value. If there are more than 1, too many values. If '
        logger.error(err)
        raise ValueError(err)

    # Resolve kwargs
    df = in_df.copy() if copy else in_df
    scorer_value: str = scorer_values[0]

    x_index, y_index, l_index, percent_filterd_per_bodypart__perc_rect = [], [], [], []

    # # Remove top row. The top row only contained project name headers
    # df_input_data_with_projectname_header_removed: pd.DataFrame = df_input_data[1:]
    # # Convert data to raw array
    # array_input_data_with_projectname_header_removed = np.array(df_input_data_with_projectname_header_removed)

    # Loop over columns, aggregate which indices in the data fall under which category.
    #   x, y, and likelihood are the three main types of columns output from DLC.
    map_back_index_to_col_name = {}
    coords_cols_names = []
    for idx_col, col in enumerate(df.columns):
        # Assign ___ TODO
        map_back_index_to_col_name[idx_col] = col
        # Columns take the regular form `FEATURE_(x|y|likelihood|coords|)`, so split by final _ OK
        column_suffix = col.split('_')[-1]
        if column_suffix == "likelihood":
            l_index.append(idx_col)
        elif column_suffix == "x":
            x_index.append(idx_col)
        elif column_suffix == "y":
            y_index.append(idx_col)
        elif column_suffix == 'coords':  # todo: delte this elif. Coords should be dropped with the io.read_csv implementation?
            # Record and check later...likely shouldn't exist anymore since its just a numbered col with no data.
            coords_cols_names.append(col)
        elif col == 'scorer': pass  # Ignore and keep 'scorer' column. It tracks the data source.
        else:
            err = f'An inappropriate column header was found: ' \
                  f'{column_suffix}. Column = "{col}". ' \
                  f'Check on CSV to see if has an unexpected output format.'
            logger.error(err)
            raise ValueError(err)
    if len(coords_cols_names) > 1:
        err = f'An unexpected number of columns were detected that contained the substring "coords". ' \
              f'Normally, there is only 1 "coords" column in a DLC output CSV, but this is an abnormal case. ' \
              f'Coords columns: {coords_cols_names} / df.head(): {df.head().to_string()}'
        logger.error(err)
        raise ValueError(err)

    # Sanity check
    if len(coords_cols_names) > 0: raise ValueError(f'coords should not exist anymore')

    # Slice data into separate arrays based on column names (derived earlier from the respective index)
    data_x: np.ndarray = np.array(df.iloc[:, np.array(x_index)])
    data_y: np.ndarray = np.array(df.iloc[:, np.array(y_index)])
    data_likelihood: np.ndarray = np.array(df.iloc[:, np.array(l_index)])
    # Note: at this point, the above 3 data arrays will all have the exact same shape

    # The below variable is instantiated with same rows as total minus 1 (for reasons TBD) and
    #   with column room for x and y values (it appears as though the likelihood values disappear)
    array_data_filtered = np.zeros((data_x.shape[0], (data_x.shape[1]) * 2))  # Initialized as zeroes to be populated later  # currdf_filt: np.ndarray = np.zeros((data_x.shape[0]-1, (data_x.shape[1]) * 2))

    logger.debug(f'{inspect.stack()[0][3]}(): Computing data threshold to forward fill any sub-threshold (x,y)...')
    percent_filterd_per_bodypart__perc_rect: List = [0. for _ in range(data_likelihood.shape[1])]  # for _ in range(data_lh.shape[1]): perc_rect.append(0)

    # Loop over data and do adaptive filtering.
    # logger.debug(f'{inspect.stack()[0][3]}: Loop over data and do adaptive filtering.')
    idx_col = 0
    for idx_col_i in tqdm(range(data_likelihood.shape[1]), desc='{inspect.stack()[0][3]}(): Adaptively filtering DLC feature %d...' % idx_col):
        # Get histogram of likelihood data in col_i (ignoring first row since its just labels (e.g.: [x  x  x  x ...]))
        histogram, bin_edges = np.histogram(data_likelihood[:, idx_col_i].astype(np.float))
        # Determine "rise".
        rise_arr = np.where(np.diff(histogram) >= 0)
        if isinstance(rise_arr, tuple):  # Sometimes np.where returns a tuple depending on input dims
            rise_arr = rise_arr[0]
        rise_0, rise_1 = rise_arr[0], rise_arr[1]

        # Threshold for bin_edges?
        if rise_arr[0] > 1:
            likelihood_threshold: np.ndarray = (bin_edges[rise_0] + bin_edges[rise_0 - 1]) / 2
        else:
            likelihood_threshold: np.ndarray = (bin_edges[rise_1] + bin_edges[rise_1 - 1]) / 2

        # Change data type to float because its currently string
        data_likelihood_col_i = data_likelihood[:, idx_col_i].astype(np.float)

        # Record percent filtered (for "reasons")
        percent_filterd_per_bodypart__perc_rect[idx_col_i] = np.sum(data_likelihood_col_i < likelihood_threshold) / data_likelihood.shape[0]

        # Note: the slicing below is just slicing the x and y columns.
        for i in range(data_likelihood.shape[0] - 1):  # TODO: low: rename `i`
            if data_likelihood_col_i[i + 1] < likelihood_threshold:
                array_data_filtered[i + 1, (2 * idx_col_i):(2 * idx_col_i + 2)] = \
                    array_data_filtered[i, (2 * idx_col_i):(2 * idx_col_i + 2)]
            else:
                array_data_filtered[i + 1, (2 * idx_col_i):(2 * idx_col_i + 2)] = \
                    np.hstack([data_x[i, idx_col_i], data_y[i, idx_col_i]])

    ### Adaptive filtering is all done. Clean up and return.
    # Remove first row in data array (values are all zeroes)
    array_filtered_data_without_first_row = np.array(array_data_filtered[1:]).astype(np.float)

    # Create DataFrame with columns by looping over x/y indices.
    columns_ordered: List[str] = []
    for x_idx, y_idx in zip(x_index, y_index):
        columns_ordered += [map_back_index_to_col_name[x_idx], map_back_index_to_col_name[y_idx]]

    # Create frame, replace 'scorer' column. Return.
    df_adaptively_filtered_data = pd.DataFrame(array_filtered_data_without_first_row, columns=columns_ordered)
    df_adaptively_filtered_data['scorer'] = scorer_value

    return df_adaptively_filtered_data, percent_filterd_per_bodypart__perc_rect


def engineer_7_features_dataframe(df: pd.DataFrame, features_names_7: List[str] = ['DistFrontPawsTailbaseRelativeBodyLength','DistBackPawsBaseTailRelativeBodyLength','InterforepawDistance','BodyLength','SnoutToTailbaseChangeInAngle','SnoutSpeed','TailbaseSpeed', ], map_names: dict = None, copy: bool = False, win_len: int = None) -> pd.DataFrame:
    # TODO: high: keep scorer col?  <----------------------------------------------------------------------------------------------------------------****
    # TODO: review https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
    #   Computationally intensive.
    """ *NEW*

    Note: you'll end up with 1 less row than you started with on input
    :param features_names_7:  TODO?
    :param win_len: TODO
    :param df: (DataFrame)
    :param map_names (dict)
    :param copy:
    :return: (DataFrame)
    """
    if win_len is None:
        win_len = original_feature_extraction_win_len_formula(config.VIDEO_FPS)

    ###

    # Args checks
    check_arg.ensure_type(df, pd.DataFrame)

    # Check for required columns
    set_df_columns = set(df.columns)
    required_config_features = [
        'SNOUT/HEAD',
        'LEFT_SHOULDER/FOREPAW',
        'RIGHT_SHOULDER/FOREPAW',
        'LEFT_HIP/HINDPAW',
        'RIGHT_HIP/HINDPAW',
        'TAILBASE',
    ]
    for feature in required_config_features:
        feature_x, feature_y = f'{config.get_part(feature)}_x', f'{config.get_part(feature)}_y'
        if feature_x not in set_df_columns:
            err_feature_x_missing = f'`{feature_x}` is required for this feature ' \
                                    f'engineering but was not found. All submitted columns are: {df.columns}'
            logger.error(err_feature_x_missing)
            raise ValueError(err_feature_x_missing)
        if feature_y not in set_df_columns:
            err_feature_y_missing = f'`{feature_y}` is required for this feature ' \
                                    f'engineering but was not found. All submitted columns are: {df.columns}'
            logger.error(err_feature_y_missing)
            raise ValueError(err_feature_y_missing)
    if 'scorer' in df.columns:
        unique_scorers = np.unique(df['scorer'].values)
        if len(unique_scorers) != 1:
            err = f'More than one scorer value found. Expected only 1. Scorer values: {unique_scorers}'
            logger.error(err)
            raise ValueError(err)
        scorer = unique_scorers[0]
    else:
        scorer = None

    # Solve kwargs

    # Do

    num_data_rows: int = len(df)
    left_shoulder_x = f'{config.get_part("LEFT_SHOULDER/FOREPAW")}_x'
    left_shoulder_y = f'{config.get_part("left_shoulder/forepaw")}_y'
    right_shoulder_x = f'{config.get_part("RIGHT_SHOULDER/FOREPAW")}_x'
    right_shoulder_y = f'{config.get_part("Right_shoulder/forepaw")}_y'
    left_hip_x = f'{config.get_part("LEFT_HIP/HINDPAW")}_x'
    left_hip_y = f'{config.get_part("LEFT_HIP/HINDPAW")}_y'
    right_hip_x = f'{config.get_part("RIGHT_HIP/HINDPAW")}_x'
    right_hip_y = f'{config.get_part("RIGHT_HIP/HINDPAW")}_y'
    tailbase_x = f'{config.get_part("Tailbase")}_x'
    tailbase_y = f'{config.get_part("Tailbase")}_y'
    head_x = f'{config.get_part("SNOUT/HEAD")}_x'
    head_y = f'{config.get_part("SNOUT/HEAD")}_y'

    ####################################################################################################################
    # Create intermediate variables to solve for final features.

    # fpd
    inter_forepaw_distance = df[[left_shoulder_x, left_shoulder_y]].values - df[[right_shoulder_x, right_shoulder_y]].values  # Previously: 'fpd'

    # cfp
    cfp__center_between_forepaws = np.vstack((
        (df[left_shoulder_x].values + df[right_shoulder_x].values) / 2,
        (df[left_shoulder_y].values + df[right_shoulder_y].values) / 2,
    )).T  # Previously: cfp

    # chp
    chp__center_between_hindpaws = np.vstack((
        (df[left_hip_x].values + df[right_hip_x].values) / 2,
        (df[left_hip_y].values + df[right_hip_y].values) / 2,
    )).T
    # cfp_pt
    dFT__cfp_pt__center_between_forepaws__minus__proximal_tail = np.vstack(([
        cfp__center_between_forepaws[:, 0] - df[tailbase_x].values,
        cfp__center_between_forepaws[:, 1] - df[tailbase_y].values,
    ])).T

    # chp_pt
    chp__center_between_hindpaws__minus__proximal_tail = np.vstack(([
        chp__center_between_hindpaws[:, 0] - df[tailbase_x].values,
        chp__center_between_hindpaws[:, 1] - df[tailbase_y].values,
    ])).T  # chp_pt

    # sn_pt
    snout__proximal_tail__distance__aka_BODYLENGTH = np.vstack(([
        df[head_x].values - df[tailbase_x].values,
        df[head_y].values - df[tailbase_y].values,
    ])).T  # previously: sn_pt

    ### Create the 4 static measurement features
    inter_forepaw_distance__normalized = np.zeros(num_data_rows)  # originally: fpd_norm
    cfp_pt__center_between_forepaws__minus__proximal_tail__normalized = np.zeros(num_data_rows)  # originally: cfp_pt_norm
    chp__proximal_tail__normalized = np.zeros(num_data_rows)  # originally: chp_pt_norm
    snout__proximal_tail__distance__aka_BODYLENGTH__normalized = np.zeros(num_data_rows)  # originally: sn_pt_norm

    for j in range(1, num_data_rows):
        # Each of these steps below produces a single-valued-array (shape: (1,1)) and inserted it into the noramlized
        inter_forepaw_distance__normalized[j] = np.array(np.linalg.norm(inter_forepaw_distance[j, :]))
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized[j] = np.linalg.norm(dFT__cfp_pt__center_between_forepaws__minus__proximal_tail[j, :])
        chp__proximal_tail__normalized[j] = np.linalg.norm(chp__center_between_hindpaws__minus__proximal_tail[j, :])
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized[j] = np.linalg.norm(snout__proximal_tail__distance__aka_BODYLENGTH[j, :])
    ## "Smooth" features for final use
    # Body length (1)
    snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed = likelihoodprocessing.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized, win_len)  # sn_pt_norm_smth
    # Inter-forepaw distance (4)
    inter_forepaw_distance__normalized__smoothed = likelihoodprocessing.boxcar_center(
        inter_forepaw_distance__normalized, win_len)  # fpd_norm_smth
    # (2)
    snout__center_forepaws__normalized__smoothed = likelihoodprocessing.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized -
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized,
        win_len)  # sn_cfp_norm_smth
    # (3)
    snout__center_hindpaws__normalized__smoothed = likelihoodprocessing.boxcar_center(
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized -
        chp__proximal_tail__normalized,
        win_len)  # sn_chp_norm_smth

    ### Create the 3 time-varying features (out of a total of 7 final features)
    snout__proximal_tail__angle = np.zeros(num_data_rows - 1)  # originally: sn_pt_ang
    snout_speed__aka_snout__displacement = np.zeros(num_data_rows - 1)  # originally: sn_disp
    tail_speed__aka_proximal_tail__displacement = np.zeros(num_data_rows - 1)  # originally: pt_disp
    for k in range(num_data_rows - 1):
        a_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k, :], 0])
        b_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :], 0])
        c = np.cross(b_3d, a_3d)
        snout__proximal_tail__angle[k] = np.dot(
            np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(
                snout__proximal_tail__distance__aka_BODYLENGTH[k, :],
                snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :])))

        snout_speed__aka_snout__displacement[k] = np.linalg.norm(
            # df[[head_x, head_y]].iloc[k + 1].values -  # TODO: IMPORTANT ******************* While this snout speed implementation matches the legacy implementation, is it really generating snout speed at all? ....
            # df[[head_x, head_y]].iloc[k].values)
            df[[head_x, ]].iloc[k + 1].values -  # TODO: IMPORTANT ******************* While this snout speed implementation matches the legacy implementation, is it really generating snout speed at all? ....
            df[[head_x, ]].iloc[k].values)

        tail_speed__aka_proximal_tail__displacement[k] = np.linalg.norm(
            # df[[tailbase_x, tailbase_y]].iloc[k+1, :].values -
            # df[[tailbase_x, tailbase_y]].iloc[k, :].values)
            df[[tailbase_x, ]].iloc[k + 1, :].values -
            df[[tailbase_x, ]].iloc[k, :].values)  # TODO: why only the x?

    snout__proximal_tail__angle__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__angle, win_len)  # sn_pt_ang_smth =>
    snout_speed__aka_snout_displacement_smoothed = likelihoodprocessing.boxcar_center(snout_speed__aka_snout__displacement, win_len)  # sn_disp_smth =>
    tail_speed__aka_proximal_tail__displacement__smoothed = likelihoodprocessing.boxcar_center(tail_speed__aka_proximal_tail__displacement, win_len)  # originally: pt_disp_smth

    # Aggregate/organize features according to original implementation
    # Note that the below features array is organized in shape: (number of features, number of records) which
    #   is typically backwards from how DataFrames are composed.
    features = np.vstack((
        snout__center_forepaws__normalized__smoothed[1:],  # 2  # TODO: problems
        snout__center_hindpaws__normalized__smoothed[1:],  # 3
        inter_forepaw_distance__normalized__smoothed[1:],  # 4
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed[1:],  # 1
        # time-varying features
        snout__proximal_tail__angle__smoothed[:],  # 7
        snout_speed__aka_snout_displacement_smoothed[:],  # 5
        tail_speed__aka_proximal_tail__displacement__smoothed[:],  # 6
    ))
    # Create DataFrame for features. Flip the features so that the records run along the rows and the
    #   features are in the columns.
    features_for_dataframe = features.T
    results_cols = features_names_7

    df_engineered_features = pd.DataFrame(features_for_dataframe, columns=results_cols)

    return df_engineered_features


def integrate_df_feature_into_100ms_bins_NEW(df, feature: str, copy: bool = False, fps=config.VIDEO_FPS) -> pd.DataFrame:
    """
    Use old algorithm to integrate features
    :param df:
    :param feature:
    :param copy: (bool)
    :param fps:
    :return:
    """
    # Arg checking
    if feature not in df.columns:
        err = f'{inspect.stack()[0][3]}(): TODO: feature not found. Cannot integrate into 100ms bins.'  # TODO
        logger.error(err)
        raise ValueError(err)
    # Kwarg resolution
    df = df.copy() if copy else df

    # Do
    # TODO  # raise NotImplementedError(f'{inspect.stack()[0][3]}(): needs to be implemented')

    return df


def train_TSNE_sklearn(df, params) -> pd.DataFrame:


    return
########################################################################################################################

def original_feature_extraction_win_len_formula(fps: int):
    """"""
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    return win_len


def extract_7_features_bsoid_tsne_py(list_of_arrays_data: List[np.ndarray], bodyparts: dict = config.BODYPARTS_PY_LEGACY, fps: int = config.VIDEO_FPS, win_len: int = None, **kwargs) -> List[np.ndarray]:
    """  * Legacy * (original implementation source: bsoid_py:bsoid_tsne()

    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions

    :param list_of_arrays_data: list of 2D arrays. Each array is processed data from a different DLC CSV output.
    :param bodyparts: dict, body parts with their orders in config
    :param fps: scalar, argument specifying camera frame-rate in config
    :param comp: int (0 or 1), argument to compile data or not (set in config)
    :param win_len:
    :return
        f_10fps: 2D array, features
        f_10fps_sc: 2D array, standardized features
        trained_tsne: 2D array, trained t-SNE space
    """
    # *note* Sometimes data is (incorrectly) submitted as an array of arrays.
    #   The number of arrays in the overarching array or, if correctly typed, list is the same # of
    #   CSV files read in. Fix type then continue.
    if isinstance(list_of_arrays_data, np.ndarray):
        warn = f'{inspect.stack()[0][3]}(): TODO: expand on this warning. Input was expected to be a ' \
               f'list of arrays but instead found array of arrays. Calling function: {inspect.stack()[1][3]}()'
        logger.warning(warn)
        list_of_arrays_data = list(list_of_arrays_data)
    # Check args
    check_arg.ensure_type(list_of_arrays_data, list)
    check_arg.ensure_type(list_of_arrays_data[0], np.ndarray)

    if not isinstance(fps, int): raise TypeError(f'fps is not integer. value = {fps}, type={type(fps)}')

    if win_len is None:
        win_len = original_feature_extraction_win_len_formula(fps)

    ### Do

    features: List[np.ndarray] = []

    # Iterate over data arrays available and build features
    for i, data_array in enumerate(list_of_arrays_data):  # for i in range(len(list_of_arrays_data)):
        logger.info(f'Extracting features from CSV file {i+1}...')
        num_data_rows = len(data_array)
        """ DELETE THIS STRING
        7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
            1. body length (or "[d_ST]"): distance from snout to base of tail
            2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT], where [d_FT] is the distance between front paws and base of tail
            3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
            4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

            5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
            6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
            7. snout to base-of-tail change in angle:

        Author also specifies that: the features are also smoothed over, or averaged across,
            a sliding window of size equivalent to 60ms (30ms prior to and after the frame of interest).
        """

        # Create some intermediate features first
        inter_forepaw_distance = data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] : 2 * bodyparts['Forepaw/Shoulder1'] + 2] - data_array[:, 2 * bodyparts['Forepaw/Shoulder2']:2 * bodyparts['Forepaw/Shoulder2'] + 2]  # Originally: 'fpd'
        cfp__center_between_forepaws = np.vstack((
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2']]) / 2,
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2'] + 1]) / 2  # TODO: RECHECK THIS LINE
        )).T  # Originally: cfp

        dFT__cfp_pt__center_between_forepaws__minus__proximal_tail = np.vstack(([
            cfp__center_between_forepaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            cfp__center_between_forepaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: cfp_pt

        chp__center_between_hindpaws = np.vstack((
            ( (data_array[:, 2 * bodyparts['Hindpaw/Hip1']] + data_array[:, 2 * bodyparts['Hindpaw/Hip2']]) / 2 ),
            ( (data_array[:, 2 * bodyparts['Hindpaw/Hip1'] + 1] + data_array[:, 2 * bodyparts['Hindpaw/Hip2'] + 1]) / 2 ),
        )).T  # Originally: chp
        chp__center_between_hindpaws__minus__proximal_tail = np.vstack(([
            chp__center_between_hindpaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            chp__center_between_hindpaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: chp_pt
        snout__proximal_tail__distance__aka_BODYLENGTH = np.vstack(([
            data_array[:, 2 * bodyparts['Snout/Head']] - data_array[:, 2 * bodyparts['Tailbase']],
            data_array[:, 2 * bodyparts['Snout/Head'] + 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: sn_pt

        ### Create the 4 static measurement features for final use ###
        inter_forepaw_distance__normalized = np.zeros(num_data_rows)  # Originally: fpd_norm
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized = np.zeros(num_data_rows)  # Originally: cfp_pt_norm
        chp__proximal_tail__normalized = np.zeros(num_data_rows)  # Originally: chp_pt_norm
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized = np.zeros(num_data_rows)  # Originally: sn_pt_norm
        for j in range(1, num_data_rows):
            inter_forepaw_distance__normalized[j] = np.array(np.linalg.norm(inter_forepaw_distance[j, :]))
            cfp_pt__center_between_forepaws__minus__proximal_tail__normalized[j] = np.linalg.norm(
                dFT__cfp_pt__center_between_forepaws__minus__proximal_tail[j, :])
            chp__proximal_tail__normalized[j] = np.linalg.norm(chp__center_between_hindpaws__minus__proximal_tail[j, :])
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized[j] = np.linalg.norm(
                snout__proximal_tail__distance__aka_BODYLENGTH[j, :])
        ## "Smooth" features for final use
        # Body length (1)
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized, win_len)  # Originally: sn_pt_norm_smth
        # Inter-forepaw distance (4)
        inter_forepaw_distance__normalized__smoothed = likelihoodprocessing.boxcar_center(inter_forepaw_distance__normalized, win_len)  # Originally: fpd_norm_smth
        # (2)
        snout__center_forepaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - cfp_pt__center_between_forepaws__minus__proximal_tail__normalized, win_len)  # Originally: sn_cfp_norm_smth
        # (3)
        snout__center_hindpaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - chp__proximal_tail__normalized, win_len)  # Originally: sn_chp_norm_smth

        ### Create the 3 time-varying features for final use ###
        snout__proximal_tail__angle = np.zeros(num_data_rows - 1)  # Originally: sn_pt_ang
        snout_speed__aka_snout__displacement = np.zeros(num_data_rows - 1)  # Originally: sn_disp
        tail_speed__aka_proximal_tail__displacement = np.zeros(num_data_rows - 1)  # Originally: pt_disp
        for k in range(num_data_rows - 1):
            b_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :], 0])
            a_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k, :], 0])
            c = np.cross(b_3d, a_3d)
            snout__proximal_tail__angle[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(snout__proximal_tail__distance__aka_BODYLENGTH[k, :], snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :])))
            snout_speed__aka_snout__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1] - data_array[k, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1])
            tail_speed__aka_proximal_tail__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1] - data_array[k, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1])
        # Smooth time-varying features
        snout__proximal_tail__angle__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__angle, win_len)  # Originally: sn_pt_ang_smth
        snout_speed__aka_snout_displacement_smoothed = likelihoodprocessing.boxcar_center(snout_speed__aka_snout__displacement, win_len)  # Originally: sn_disp_smth
        tail_speed__aka_proximal_tail__displacement__smoothed = likelihoodprocessing.boxcar_center(tail_speed__aka_proximal_tail__displacement, win_len)  # Originally: pt_disp_smth

        # Append final features to features list
        features.append(np.vstack((  # Do not change order unless you know what you're doing
            # static features
            snout__center_forepaws__normalized__smoothed[1:],                           # 2
            snout__center_hindpaws__normalized__smoothed[1:],                           # 3
            inter_forepaw_distance__normalized__smoothed[1:],                           # 4
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed[1:],    # 1
            # time-varying features
            snout__proximal_tail__angle__smoothed[:],                                   # 7
            snout_speed__aka_snout_displacement_smoothed[:],                            # 5
            tail_speed__aka_proximal_tail__displacement__smoothed[:],                   # 6
        )))
        # End loop // Loop to next data_array
    # Exit
    logger.debug(f'{inspect.stack()[0][3]}(): Done extracting features from a '
                 f'total of {len(list_of_arrays_data)} training CSV files.')

    return features


def integrate_features_into_100ms_bins_LEGACY(data: List[np.ndarray], features: List[np.ndarray], fps: int = config.VIDEO_FPS) -> List[np.ndarray]:
    """ * Legacy *
    TODO
    :param data: (list of arrays) raw data? TODO
    :param features: (list of arrays) extracted features? TODO
    :param fps:
    :return:
    """
    fps_divide_10 = round(fps / 10)
    features_10fps: List[np.ndarray] = []

    for n, features_n in enumerate(features):
        features_100ms_n = np.zeros(len(data[n]))
        for k in range(fps_divide_10 - 1, len(features_n[0]), fps_divide_10):
            if k > fps_divide_10 - 1:
                features_100ms_n = np.concatenate((
                    features_100ms_n.reshape(features_100ms_n.shape[0], features_100ms_n.shape[1]),
                    np.hstack((np.mean((features_n[0:4, range(k - fps_divide_10, k)]), axis=1),
                               np.sum((features_n[4:7, range(k - fps_divide_10, k)]), axis=1)))
                    .reshape(len(features[0]), 1)), axis=1)
            else:
                features_100ms_n = np.hstack((
                    np.mean((features_n[0:4, range(k - fps_divide_10, k)]), axis=1),
                    np.sum((features_n[4:7, range(k - fps_divide_10, k)]), axis=1),
                )).reshape(len(features[0]), 1)

        features_10fps.append(features_100ms_n)
        logger.debug(f'{inspect.stack()[0][3]}(): Done integrating features into 100ms bins from CSV file #{n+1}.')
    return features_10fps
