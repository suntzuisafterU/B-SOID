"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.

Potential abbreviations:
    sn: snout
    pt: proximal tail ?
"""
# Hierarchical Density-Based Spatial Clustering of Applications with Noise
from bhtsne import tsne as TSNE_bthsne
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, List, Tuple
import hdbscan
import inspect
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import umap

from bsoid import config, feature_engineering, visuals, check_arg
from bsoid.util import likelihoodprocessing

logger = config.initialize_logger(__name__)


### NEW ###


@config.deco__log_entry_exit(logger)
def train_TSNE_NEW(list_of_arrays_data: pd.DataFrame, features, fps=config.VIDEO_FPS, dimensions: int = 3, comp=None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, TSNE_bthsne, StandardScaler]:
    """
    Encapsulates legacy implementation of training TSNE

    :param list_of_arrays_data:
    :param features:
    :param fps:
    :param dimensions:
    :return:
    """
    features_10fps = None

    # What is it doing here????
    features_10fps = feature_engineering.integrate_into_bins(list_of_arrays_data, features)

    # Scale data, xform
    scaler_obj = StandardScaler()
    scaler_obj.fit(features_10fps.T)
    f_10fps_scaled = scaler_obj.transform(features_10fps.T).T
    # Train TSNE
    logger.debug(f'{inspect.stack()[0][3]}:Training t-SNE to embed {f_10fps_scaled.shape[1]} instances'
                 f'from {f_10fps_scaled.shape[0]} D into 3 D from a total of {len(list_of_arrays_data)} CSV files...')

    trained_tsne = TSNE_bthsne(
        f_10fps_scaled.T,
        dimensions=dimensions,
        perplexity=np.sqrt(f_10fps_scaled.shape[1]),
        theta=0.5,
        rand_seed=config.RANDOM_STATE,
        # TODO: high: add verbose options
    )

    logger.debug(f'{inspect.stack()[0][3]}::Done embedding into 3 D.')
    return features_10fps, f_10fps_scaled, trained_tsne, scaler_obj


def train_TSNE_LEGACY(list_of_arrays_data: List[np.ndarray], features: List[np.ndarray], fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING) -> Tuple[Any, Any, Any, StandardScaler]:
    """
    Encapsulates legacy implementation of training TSNE

    :param list_of_arrays_data:
    :param features:
    :param fps:
    :param comp:
    :return:
    """
    # 2/2 train TSNE
    if comp == 0:
        features_10fps = []
        f_10fps_scaled = []
        trained_tsne = []
    # loop over data, put into 100ms bins
    for n, feature_n in enumerate(features):  # for n in range(len(features)):
        feats1 = np.zeros(len(list_of_arrays_data[n]))
        for k in range(round(fps / 10) - 1, len(feature_n[0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feature_n[0:4, range(k - round(fps / 10), k)]), axis=1), np.sum((feature_n[4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feature_n[0:4, range(k - round(fps / 10), k)]), axis=1), np.sum((feature_n[4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)
        logger.info(f'{inspect.stack()[0][3]}: Done integrating features into 100ms bins from CSV file {n+1}.')

        if comp == 1:
            if n > 0: features_10fps = np.concatenate((features_10fps, feats1), axis=1)
            else: features_10fps = feats1
        elif comp == 0:
            features_10fps.append(feats1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T).T
            f_10fps_scaled.append(feats1_stnd)
            logger.info(f'Training t-SNE to embed {f_10fps_scaled[n].shape[1]} instances from '
                        f'{f_10fps_scaled[n].shape[0]} D into 3 D from CSV file {n + 1}...')
            trained_tsne_i = TSNE_bthsne(
                f_10fps_scaled[n].T,
                dimensions=3, perplexity=np.sqrt(f_10fps_scaled[n].shape[1]),
                theta=0.5, rand_seed=config.RANDOM_STATE
            )
            trained_tsne.append(trained_tsne_i)
            logger.info('Done embedding into 3 D.')
        else:
            err = f'non-valid comp value found. Value = {comp}'
            logger.error(err)
            raise ValueError(err)
    if comp == 1:
        scaler: StandardScaler = StandardScaler()
        scaler.fit(features_10fps.T)  # TODO: HIGH: variable `f_10fps` referenced before assignment. Error in logic above? ########################## IMPORTANT ###############################
        f_10fps_scaled = scaler.transform(features_10fps.T).T
        logger.info(f'{inspect.stack()[0][3]}:Training t-SNE to embed {f_10fps_scaled.shape[1]} instances'
                    f'from {f_10fps_scaled.shape[0]} D into 3 D from a total of {len(list_of_arrays_data)} CSV files...')
        trained_tsne = TSNE_bthsne(f_10fps_scaled.T, dimensions=3, perplexity=np.sqrt(f_10fps_scaled.shape[1]),
                                   theta=0.5, rand_seed=config.RANDOM_STATE)
        logger.info(f'{inspect.stack()[0][3]}::Done embedding into 3 D.')
    return features_10fps, f_10fps_scaled, trained_tsne, scaler


@config.deco__log_entry_exit(logger)
def train_emgmm_with_learned_tsne_space_NEW(df_trained_tsne: pd.DataFrame, emgmm_params=config.EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param df_trained_tsne: (data is trained t-sne space)
    :param emgmm_params: (Dict)
    :return assignments: Converged EM-GMM group assignments
    """
    num_rows, num_columns = len(df_trained_tsne), len(df_trained_tsne.columns)
    logger.debug(f'Running EM-GMM on {num_rows} instances in {num_columns} D space...')
    clf_gmm = GaussianMixture(**emgmm_params).fit(df_trained_tsne)

    logger.debug(f'Predicting labels for {num_rows} instances '
                 f'in {num_columns} D space...')

    assignments = clf_gmm.predict(df_trained_tsne)

    logger.debug(f'Done predicting labels for {num_rows} instances '
                 f'in {num_columns} D space...')

    uk = list(np.unique(assignments))
    assignments_list: List[int] = []
    for assignment in assignments:
        assignment_idx_value = uk.index(assignment)
        assignments_list += [assignment_idx_value]
    assignments: np.ndarray = np.array(assignments_list)

    # Sanity check
    if len(df_trained_tsne) != assignments.shape[0]:
        # TODO: remove this err check after debugging efforts?
        err = f'The number of assignments doesnt match the total number of input entries'
        logger.error(err)
        raise ValueError(err)

    return assignments


@config.deco__log_entry_exit(logger)
def train_SVM__bsoid_svm_py(df, features_list: List[str], label: str, features: np.ndarray, labels: np.ndarray, comp: int = config.COMPILE_CSVS_FOR_TRAINING, holdout_pct: float = config.HOLDOUT_PERCENT, cv_it: int = config.CROSSVALIDATION_K, svm_params: dict = config.SVM_PARAMS) -> SVC:
    """
    Train SVM classifier
    :param comp:
    :param features: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param holdout_pct: (float) Test partition ratio for validating SVM performance in GLOBAL_CONFIG
    :param cv_it: (int) iterations for cross-validation in GLOBAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    # Args checking
    check_arg.ensure_type(features_list, list)
    check_arg.ensure_type(label, str)
    set_df_columns = set(df.columns)
    for feature in features_list + [label, ]:
        if feature not in set_df_columns:
            feature_missing_err = f'{inspect.stack()[0][3]}(): {feature} is ' \
                                  f'missing for SVM training. Columns: {df.columns}. TODO: Make msg better.'
            logger.error(feature_missing_err)
            raise ValueError(feature_missing_err)
    df_features, df_label = df[features_list], df[label]

    feats_train, feats_test, labels_train, labels_test = train_test_split(
        df_features, df_label, test_size=holdout_pct, random_state=config.RANDOM_STATE)

    logger.info(f'Training SVM on randomly partitioned {(1-holdout_pct)*100}% of training data...')

    classifier = SVC(**svm_params)
    classifier.fit(df[features_list], df[label])

    logger.debug(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
    logger.debug(f'Predicting randomly sampled (non-overlapped) assignments '
                 f'using the remaining {holdout_pct * 100}%...')

    # TODO: extract the plotting of graphs from in this function
    if config.PLOT_GRAPHS:
        # Plot SVM accuracy

        # Plot confusion matrix
        # np.set_printoptions(precision=2)  # TODO: low: address this line later
        titles_options = [('Non-normalized confusion matrix', None, 'counts'),
                          ('Normalized confusion matrix', 'true', 'normalized'), ]
        for title, normalize, norm_label in titles_options:
            display = plot_confusion_matrix(classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
            display.ax_.set_title(title)
            print(title, display.confusion_matrix, sep='\n')
            if config.SAVE_GRAPHS_TO_FILE:
                file_name = f'confusion_matrix_{norm_label}_{config.runtime_timestr}'
                visuals.save_graph_to_file(display.figure_, file_name)
        plt.show()

    logger.info(f'{inspect.stack()[0][3]}(): Scored cross-validated SVM performance.')  # Previously: .format(feats_train.shape, labels_train.shape))

    return classifier


########################################################################################################################
# @config.deco__log_entry_exit(logger)
def extract_7_features_bsoid_tsne_py(list_of_arrays_data: List[np.ndarray], bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING) -> List[np.ndarray]:
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions

    * DOES NOT integrate features into 100ms bins *

    :param list_of_arrays_data: list of 3D array
    :param bodyparts: dict, body parts with their orders in config
    :param fps: scalar, argument specifying camera frame-rate in config
    :param comp: boolean (0 or 1), argument to compile data or not in config
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    replacement_func = feature_engineering.extract_7_features_bsoid_tsne_py
    deprec_warning = f'{inspect.stack()[0][3]}(): This function will be deprecated in ' \
                     f'favour of the new module/func feature_engineering."{replacement_func.__qualname__}()" soon. ' \
                     f'Caller = {inspect.stack()[1][3]}'
    logger.warning(deprec_warning)

    ### *note* Sometimes data is (incorrectly) submitted as an array of arrays (the number
    # of arrays in the top-level array or, if correctly typed, list) is the same number of
    # CSV files read in). Fix type then continue.
    if isinstance(list_of_arrays_data, np.ndarray):
        logger.warning(f'TODO: expand on warning: list of arrays was expected to be a list but instead found np array')  # TODO: expand on warning
        list_of_arrays_data = list(list_of_arrays_data)
    # Check args
    if len(list_of_arrays_data) == 0:
        err_empty_list_input = f'TODO: empty list input'  # TODO
        logger.error(err_empty_list_input)
        raise ValueError(err_empty_list_input)
    for arr in list_of_arrays_data:
        check_arg.ensure_type(arr, np.ndarray)
    check_arg.ensure_type(list_of_arrays_data, list)

    ###
    win_len = feature_engineering.original_feature_extraction_win_len_formula(fps)  # win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features: List[np.ndarray] = []

    # Iterate over data arrays available and build features
    for i, data_array in enumerate(list_of_arrays_data):  # for i in range(len(list_of_arrays_data)):
        logger.debug(f'{inspect.stack()[0][3]}(): Extracting features from CSV file {i+1}...')
        num_data_rows = len(data_array)
        #
        # inter_forepaw_distance resultant shape is (all rows, 2 columns)
        inter_forepaw_distance = data_array[:, 2 * bodyparts['Forepaw/Shoulder1']:2 * bodyparts['Forepaw/Shoulder1'] + 2] - data_array[:, 2 * bodyparts['Forepaw/Shoulder2']:2 * bodyparts['Forepaw/Shoulder2'] + 2]  # Previously: 'fpd'

        cfp__center_between_forepaws = np.vstack((
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2']]) / 2,
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1] + data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1]) / 2),
        ).T  # Previously: cfp

        dFT__cfp_pt__center_between_forepaws__minus__proximal_tail = np.vstack(([
            cfp__center_between_forepaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            cfp__center_between_forepaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Previously: cfp_pt
        chp__center_between_hindpaws = np.vstack((
            ((data_array[:, 2 * bodyparts['Hindpaw/Hip1']] + data_array[:, 2 * bodyparts['Hindpaw/Hip2']]) / 2),
            ((data_array[:, 2 * bodyparts['Hindpaw/Hip1'] + 1] + data_array[:, 2 * bodyparts['Hindpaw/Hip2'] + 1]) / 2),
        )).T
        chp__center_between_hindpaws__minus__proximal_tail = np.vstack(([
            chp__center_between_hindpaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            chp__center_between_hindpaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  #  chp_pt
        snout__proximal_tail__distance__aka_BODYLENGTH = np.vstack(([
            data_array[:, 2 * bodyparts['Snout/Head']] - data_array[:, 2 * bodyparts['Tailbase']],
            data_array[:, 2 * bodyparts['Snout/Head'] + 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # previously: sn_pt

        ### Create the 4 static measurement features ###
        inter_forepaw_distance__normalized = np.zeros(num_data_rows)        # originally: fpd_norm
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized = np.zeros(num_data_rows) # originally: cfp_pt_norm
        chp__proximal_tail__normalized = np.zeros(num_data_rows)            # originally: chp_pt_norm
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized = np.zeros(num_data_rows)  # originally: sn_pt_norm
        for j in range(1, num_data_rows):
            # Each of these steps below produces a single-valued-array (shape: (1,1)) and inserted it into the noramlized
            inter_forepaw_distance__normalized[j] = np.array(np.linalg.norm(inter_forepaw_distance[j, :]))
            cfp_pt__center_between_forepaws__minus__proximal_tail__normalized[j] = np.linalg.norm(dFT__cfp_pt__center_between_forepaws__minus__proximal_tail[j, :])
            chp__proximal_tail__normalized[j] = np.linalg.norm(chp__center_between_hindpaws__minus__proximal_tail[j, :])
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized[j] = np.linalg.norm(snout__proximal_tail__distance__aka_BODYLENGTH[j, :])
        ## "Smooth" features for final use
        # Body length (1)
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized, win_len)                           # sn_pt_norm_smth
        # Inter-forepaw distance (4)
        inter_forepaw_distance__normalized__smoothed = likelihoodprocessing.boxcar_center(inter_forepaw_distance__normalized, win_len)                                               # fpd_norm_smth
        # (2)
        snout__center_forepaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - cfp_pt__center_between_forepaws__minus__proximal_tail__normalized, win_len)   # sn_cfp_norm_smth
        # (3)
        snout__center_hindpaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - chp__proximal_tail__normalized, win_len)   # sn_chp_norm_smth

        ### Create the 3 time-varying features ###
        snout__proximal_tail__angle = np.zeros(num_data_rows - 1)                   # originally: sn_pt_ang
        snout_speed__aka_snout__displacement = np.zeros(num_data_rows - 1)          # originally: sn_disp
        tail_speed__aka_proximal_tail__displacement = np.zeros(num_data_rows - 1)   # originally: pt_disp
        for k in range(num_data_rows - 1):
            b_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :], 0])
            a_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k, :], 0])
            c = np.cross(b_3d, a_3d)
            snout__proximal_tail__angle[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(snout__proximal_tail__distance__aka_BODYLENGTH[k, :], snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :])))
            snout_speed__aka_snout__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1] - data_array[k, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1])
            tail_speed__aka_proximal_tail__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1] - data_array[k,2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1])
        # Smooth features for final use
        snout__proximal_tail__angle__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__angle, win_len)  # sn_pt_ang_smth =>
        snout_speed__aka_snout_displacement_smoothed = likelihoodprocessing.boxcar_center(snout_speed__aka_snout__displacement, win_len)  # sn_disp_smth =>
        tail_speed__aka_proximal_tail__displacement__smoothed = likelihoodprocessing.boxcar_center(tail_speed__aka_proximal_tail__displacement, win_len)  # originally: pt_disp_smth

        ### Append final features to features list ###
        features.append(np.vstack((
            snout__center_forepaws__normalized__smoothed[1:],                           # 2
            snout__center_hindpaws__normalized__smoothed[1:],                           # 3
            inter_forepaw_distance__normalized__smoothed[1:],                           # 4
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed[1:],    # 1
            # time-varying features
            snout__proximal_tail__angle__smoothed[:],                                   # 7
            snout_speed__aka_snout_displacement_smoothed[:],                            # 5
            tail_speed__aka_proximal_tail__displacement__smoothed[:],)                  # 6
        ))
        # Loop to next data_array
    # Exit
    logger.debug(f'{inspect.stack()[0][3]}: Done extracting features from a '
                 f'total of {len(list_of_arrays_data)} training CSV files.')

    return features




def extract_features_and_train_TSNE(list_of_arrays_data: List[np.ndarray], bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING) -> Tuple[Any, Any, Any, Any]:
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions

    :param list_of_arrays_data: list of 3D array
    :param bodyparts: dict, body parts with their orders in LOCAL_CONFIG
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    # 1/2 Extract features (Not including pushing to 100 ms bins. Check legacy implementation to see if that was part of the deal)
    features = feature_engineering.extract_7_features_bsoid_tsne_py(list_of_arrays_data, bodyparts=bodyparts, fps=fps, comp=comp)

    # 2/2 Train tsne
    # features_10fps, f_10fps_scaled, trained_tsne, scaler = train_TSNE_LEGACY(list_of_arrays_data, features, fps=fps, comp=comp)
    features_10fps, f_10fps_scaled, trained_tsne, scaler = train_TSNE_NEW(list_of_arrays_data, features, fps=fps)


    return features_10fps, f_10fps_scaled, trained_tsne, scaler


########################################################################################################################

@config.deco__log_entry_exit(logger)
def train_umap_unsupervised_with_xy_features_umapapp(data: List[np.ndarray], fps: int = config.VIDEO_FPS) -> Tuple:
    # TODO: high: ensure that the final logic matches original functions..ensure no renaming side-effects occurred
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :return (tuple):
        f_10fps: 2D array, features
        f_10fps_sc: 2D array, standardized/session features
    """
    win_len = feature_engineering.original_feature_extraction_win_len_formula(fps)  # win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logger.debug(f'{inspect.stack()[0][3]}:Extracting features from CSV file {m + 1}...')
        data_range = len(data[m])
        dis_r, dxy_r = [], []
        for r in range(data_range):
            if r < data_range - 1:
                dis = []
                for c in range(data[m].shape[1], 2):
                    dis.append(np.linalg.norm(data[m][r + 1, c:c + 2] - data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(data[m].shape[1], 2), 2):
                dxy.append(data[m][r, i:i + 2] - data[m][r, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)

        ang_smth, dis_smth, dxy_smth = [], [], []
        dxy_eu = np.zeros([data_range, dxy_r.shape[1]])
        ang = np.zeros([data_range - 1, dxy_r.shape[1]])
        for l in range(dis_r.shape[1]):
            dis_smth.append(likelihoodprocessing.boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(data_range):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < data_range - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(likelihoodprocessing.boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(likelihoodprocessing.boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        features.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))  # TODO: med: sometimes this line causes an error when dx_smth is indexed (IndexError: too many indicies for array)
    logger.info(f'Done extracting features from a total of {len(data)} training CSV files.')

    for n in range(len(features)):
        features_n = np.zeros(len(data[n]))
        for k in range(round(fps / 10), len(features[n][0]), round(fps / 10)):
            if k > round(fps / 10):
                features_n = np.concatenate((features_n.reshape(features_n.shape[0], features_n.shape[1]),
                                         np.hstack((np.mean((features[n][0:dxy_smth.shape[0],
                                                             range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((features[n][dxy_smth.shape[0]:features[n].shape[0],
                                                            range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                features_n = np.hstack((np.mean((features[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]), axis=1),
                                        np.sum((features[n][dxy_smth.shape[0]:features[n].shape[0],
                                                range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)
        logger.info(f'{inspect.stack()[0][3]}::Done integrating features into 100ms bins from CSV file {n+1}.')

        if n > 0:  # For any index value of n that isn't the very first run
            f_10fps = np.concatenate((f_10fps, features_n), axis=1)
            scaler = StandardScaler()
            scaler.fit(features_n.T)
            features_n_scaled = scaler.transform(features_n.T).T
            f_10fps_sc = np.concatenate((f_10fps_sc, features_n_scaled), axis=1)
        else:
            f_10fps = features_n
            scaler = StandardScaler()
            scaler.fit(features_n.T)
            features_n_scaled = scaler.transform(features_n.T).T
            f_10fps_sc = features_n_scaled  # scaling is important as I've seen wildly different stdev/feat between sessions
    logger.debug(f'{inspect.stack()[0][3]: Now exiting.}')
    return f_10fps, f_10fps_sc
@config.deco__log_entry_exit(logger)
def bsoid_umap_embed_umapapp(features_10fps_scaled, umap_params=config.UMAP_PARAMS) -> Tuple[umap.UMAP, Any]:
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param features_10fps_scaled: (originally 'f_10fps_sc') 2D array, standardized/session features
    :param umap_params: dict, UMAP params in GLOBAL_CONFIG
    :return trained_umap: object, trained UMAP transformer
    :return umap_embeddings: 2D array, embedded UMAP space
    """
    # The comment below was commented-out by the original authors, but has been kept here for continuity.
    ###### So far, use of PCA is not necessary. If, however, features go beyond 100, consider taking top 50 PCs #####
    # if f_10fps_sc.shape[0] > 50:
    #     logger.info('Compressing {} instances from {} D '
    #                  'into {} D using PCA'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
    #                                               50))
    #     feats_train = PCA(n_components=50, random_state=23).fit_transform(f_10fps_sc.T)
    #     pca = PCA(n_components=50).fit(f_10fps_sc.T)
    #     logger.info('Done linear transformation with PCA.')
    #     logger.info('The top {} Principal Components '
    #                  'explained {}% variance'.format(50, 100 * np.sum(pca.explained_variance_ratio_)))
    ################ FastICA potentially useful for demixing signal ################
    # lowd_feats = FastICA(n_components=10, random_state=23).fit_transform(f_10fps.T)
    # feats_train = lowd_feats
    features_train = features_10fps_scaled.T
    logger.debug(f'{inspect.stack()[0][3]}:Transforming all {features_train.shape[0]} instances '
                 f'from {features_train.shape[1]} D into {umap_params.get("n_components")} D')
    trained_umap = umap.UMAP(n_neighbors=int(round(np.sqrt(features_train.shape[0]))),  # Power Law
                             **{k: v for k, v in umap_params.items() if k != 'n_neighbors'}).fit(features_train)
    umap_embeddings = trained_umap.embedding_
    logger.debug(f'{inspect.stack()[0][3]}:Done non-linear transformation with UMAP from {features_train.shape[1]} D '
                 f'into {umap_embeddings.shape[1]} D.')
    logger.debug(f'{inspect.stack()[0][3]}: now exiting.')
    return trained_umap, umap_embeddings

@config.deco__log_entry_exit(logger)
def bsoid_hdbscan_umapapp(umap_embeddings, hdbscan_params=config.HDBSCAN_PARAMS) -> Tuple[Any, np.ndarray, Any]:
    """
    Trains HDBSCAN (unsupervised) given learned UMAP space
    :param umap_embeddings: 2D array, embedded UMAP space
    :param hdbscan_params: dict, HDBSCAN params in GLOBAL_CONFIG
    :return assignments: HDBSCAN assignments
    """
    highest_numulab_value = -np.infty
    numulab = []
    min_cluster_range = range(6, 21)  # TODO: Q: why is the range this way? Magic variables?
    logger.info('Running HDBSCAN on {} instances in {} D space...'.format(*umap_embeddings.shape))
    for min_c in min_cluster_range:
        # trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
        #                                      min_cluster_size=round(umap_embeddings.shape[0] * 0.007),  # just < 1%/cluster
        #                                      **hdbscan_params).fit(umap_embeddings)
        trained_classifier = hdbscan.HDBSCAN(
            prediction_data=True,
            min_cluster_size=int(round(0.001 * min_c * umap_embeddings.shape[0])),
            **hdbscan_params
        ).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab_value:
            logger.debug(f'{likelihoodprocessing.get_current_function()}(): '
                         f'Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab_value = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    logger.info(f'{likelihoodprocessing.get_current_function()}: ' +
                'Done predicting labels for {} instances in {} D space...'.format(*umap_embeddings.shape))
    return assignments, soft_clusters, soft_assignments

@config.deco__log_entry_exit(logger)
def bsoid_nn_appumap(feats, labels, holdout_pct: float = config.HOLDOUT_PERCENT, cv_it: int = config.CROSSVALIDATION_K,
                     mlp_params=config.MLP_PARAMS) -> Tuple[Any, Any, Any]:
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, HDBSCAN assignments
    :param holdout_pct: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return clf: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    :return nn_assignments: 1D array, neural net predictions
    """
    features_filtered = feats[:, labels >= 0]
    labels_filtered = labels[labels >= 0]
    feats_train, feats_test, labels_train, labels_test = train_test_split(features_filtered.T, labels_filtered.T, test_size=holdout_pct, random_state=config.RANDOM_STATE)
    logger.info(f'Training feedforward neural network on randomly '
                f'partitioned {(1-holdout_pct)*100}% of training data...')
    classifier = MLPClassifier(**mlp_params)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**mlp_params)
    clf.fit(features_filtered.T, labels_filtered.T)
    nn_assignments = clf.predict(feats.T)
    logger.info(f'Done training feedforward neural network mapping {feats_train.shape} features '
                f'to {labels_train.shape} assignments.')
    scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)

    if config.PLOT_GRAPHS:  # TODO: low: saving the plot requires the plot to be shown
        np.set_printoptions(precision=2)
        titles_options = [("Non-normalized confusion matrix", None),
                          ("Normalized confusion matrix", 'true')]
        title_names, j = ['counts', 'norm'], 0
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            if config.SAVE_GRAPHS_TO_FILE:
                file_name = f'confusion_matrix_{title_names[j]}_{config.runtime_timestr}'  # my_file = f'confusion_matrix_{title_names[j]}' # disp.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}{time_str}.svg'))
                visuals.save_graph_to_file(disp.figure_, file_name)
            j += 1
        plt.show()
    logger.info(f'Scored cross-validated feedforward neural network performance. '
                f'Features shape: {feats_train.shape} / labels shape: {labels_train.shape}')
    return clf, scores, nn_assignments

@config.deco__log_entry_exit(logger)
def train_mlp_classifier_voc(feats, labels, comp: int = config.COMPILE_CSVS_FOR_TRAINING, holdout_percent: float = config.HOLDOUT_PERCENT, crossvalidation_k: int = config.CROSSVALIDATION_K, mlp_params=config.MLP_PARAMS) -> Tuple[Any, Any]:
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param comp: TODO
    :param holdout_percent: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param crossvalidation_k: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return classifier: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    """
    # RECALL COMP: # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            feats.T, labels.T, test_size=holdout_percent, random_state=config.RANDOM_STATE)
        logger.info(f'Training feedforward neural network on randomly '
                    f'partitioned {(1-holdout_percent)*100}% of training data...')
        classifier = MLPClassifier(**mlp_params)
        classifier.fit(feats_train, labels_train)
        logger.info(f'Done training feedforward neural network mapping {feats_train.shape} features to {labels_train.shape} assignments.')
        logger.info('Predicting randomly sampled (non-overlapped) assignments '
                    f'using the remaining {holdout_percent*100}%...')
        scores = cross_val_score(classifier, feats_test, labels_test, cv=crossvalidation_k, n_jobs=config.CROSSVALIDATION_N_JOBS)
        time_str = config.runtime_timestr  # time_str = time.strftime("_%Y%m%d_%H%M")
        if config.PLOT_GRAPHS:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            file_title_names = ["raw_counts", "normalized", ]
            j = 0
            for graph_title, normalize in titles_options:
                display = plot_confusion_matrix(
                    classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
                display.ax_.set_title(graph_title)
                print(graph_title)
                print(display.confusion_matrix)
                if config.SAVE_GRAPHS_TO_FILE:
                    file_name = f'confusion_matrix__{file_title_names[j]}_{time_str}'  # display.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{file_name}{time_str}.svg'))
                    visuals.save_graph_to_file(display.figure_, file_name)
                    logger.debug(f'Saved graph titled: {file_name}')
                j += 1
            plt.show()
    elif comp == 0:
        classifier = []
        scores = []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(
                feats[i].T, labels[i].T, test_size=holdout_percent, random_state=config.RANDOM_STATE)
            logger.info(f'Training feedforward neural network on randomly partitioned {(1 - holdout_percent) * 100}% of training data...')
            clf = MLPClassifier(**mlp_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logger.info(
                f'Done training feedforward neural network mapping {feats_train.shape} features to {labels_train.shape} assignments.')
            logger.info(f'Predicting randomly sampled (non-overlapped) assignments using the remaining {holdout_percent*100}%...')
            sc = cross_val_score(classifier, feats_test, labels_test,
                                 cv=crossvalidation_k, n_jobs=config.CROSSVALIDATION_N_JOBS)  # TODO: why does this line exist? Missing usage cross validation plotting?
            time_str = time.strftime("_%Y%m%d_%H%M")
            if config.PLOT_GRAPHS:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                file_title_names = ["counts", "norm"]
                for graph_title, normalize in titles_options:
                    display = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                    cmap=plt.cm.Blues, normalize=normalize)
                    display.ax_.set_title(graph_title)
                    print(graph_title)
                    print(display.confusion_matrix)
                    if config.SAVE_GRAPHS_TO_FILE:
                        file_name = f'confusion_matrix__classifier{i}__{file_title_names[j]}_{time_str}'  # display.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{file_name}{time_str}.svg'))
                        visuals.save_graph_to_file(display.figure_, file_name)
                        logger.debug(f'Saved graph titled: {file_name}')
                    j += 1
                plt.show()
    else:
        err = f'Invalid comp valid detected: {comp}.'
        logger.error(err)
        raise ValueError(err)

    logger.info(f'Scored cross-validated feedforward neural network performance. '
                f'Features shape: {feats_train.shape} / labels shape: {labels_train.shape}')
    return classifier, scores


def bsoid_tsne_voc(data: list, bodyparts=config.BODYPARTS_VOC_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING, tsne_params=config.TSNE_SKLEARN_PARAMS) -> Tuple[List, List, np.ndarray]:
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions

    :param data: list of 3D array
    :param bodyparts: dict, body parts with their orders in LOCAL_CONFIG
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :retrun f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    if len(data) <= 0:
        err = f'`data` was expected to be list of data (specifically arrays) but no data was found. data = {data}'
        logger.error(err)
        raise ValueError(err)
    win_len = feature_engineering.original_feature_extraction_win_len_formula(fps)  # win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    logger.info(f'{inspect.stack()[0][3]}: Looping over data and extract features')
    for m in range(len(data)):
        logger.debug(f'Extracting features from CSV file {m+1} ...')
        data_range = len(data[m])
        p12 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - data[m][:, 2 * bodyparts.get('Point2'):2 * bodyparts.get('Point2') + 2]
        p14 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - data[m][:, 2 * bodyparts.get('Point4'):2 * bodyparts.get('Point4') + 2]
        p15 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point5'):2 * bodyparts.get('Point5') + 2]
        p18 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point8'):2 * bodyparts.get('Point8') + 2]

        p15_norm = np.zeros(data_range)
        p18_norm = np.zeros(data_range)
        for i in range(1, data_range):
            p15_norm[i] = np.array(np.linalg.norm(p15[i, :]))
            p18_norm[i] = np.array(np.linalg.norm(p18[i, :]))
        p15_norm_smth = likelihoodprocessing.boxcar_center(p15_norm, win_len)
        p18_norm_smth = likelihoodprocessing.boxcar_center(p18_norm, win_len)

        p12_ang = np.zeros(data_range - 1)
        p14_ang = np.zeros(data_range - 1)
        p3_disp = np.zeros(data_range - 1)
        p7_disp = np.zeros(data_range - 1)
        for k in range(data_range-1):
            b_3d = np.hstack([p12[k + 1, :], 0])
            a_3d = np.hstack([p12[k, :], 0])
            c = np.cross(b_3d, a_3d)
            p12_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(p12[k, :], p12[k + 1, :])))
            e_3d = np.hstack([p14[k + 1, :], 0])
            d_3d = np.hstack([p14[k, :], 0])
            f = np.cross(e_3d, d_3d)
            p14_ang[k] = np.dot(np.dot(np.sign(f[2]), 180) / np.pi, math.atan2(np.linalg.norm(f), np.dot(p14[k, :], p14[k + 1, :])))
            p3_disp[k] = np.linalg.norm(data[m][k + 1, 2 * bodyparts.get('Point3'):2 * bodyparts.get('Point3') + 1] - data[m][k, 2 * bodyparts.get('Point3'):2 * bodyparts.get('Point3') + 1])
            p7_disp[k] = np.linalg.norm(data[m][k + 1, 2 * bodyparts.get('Point7'):2 * bodyparts.get('Point7') + 1] - data[m][k, 2 * bodyparts.get('Point7'):2 * bodyparts.get('Point7') + 1])
        p12_ang_smth = likelihoodprocessing.boxcar_center(p12_ang, win_len)
        p14_ang_smth = likelihoodprocessing.boxcar_center(p14_ang, win_len)
        p3_disp_smth = likelihoodprocessing.boxcar_center(p3_disp, win_len)
        p7_disp_smth = likelihoodprocessing.boxcar_center(p7_disp, win_len)
        feats.append(np.vstack((p15_norm_smth[1:], p18_norm_smth[1:],
                                p12_ang_smth[:], p14_ang_smth[:], p3_disp_smth[:], p7_disp_smth[:])))
    logger.info(f'{inspect.stack()[0][3]}(): Done extracting features from a total of {len(data)} training CSV files.')
    if comp == 0:
        features_10fps = []
        features_10fps_scaled = []
        trained_tsne = []
    for n in range(len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((
                    np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]), axis=1)),
                ).reshape(len(feats[0]), 1)
        logger.debug(f'Done integrating features into 100ms bins from CSV file {n+1}.')
        if comp == 1:
            if n > 0:
                features_10fps = np.concatenate((features_10fps, feats1), axis=1)
            else:
                features_10fps = feats1
        elif comp == 0:
            features_10fps.append(feats1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T).T
            features_10fps_scaled.append(feats1_stnd)
            logger.info(f'Training t-SNE to embed {features_10fps_scaled[n].shape[1]} instances '
                        f'from {features_10fps_scaled[n].shape[0]} D '
                        f'into 3 D from CSV file {n+1}...')
            trained_tsne_i = TSNE_sklearn(perplexity=np.sqrt(features_10fps_scaled[n].shape[1]),
                                          early_exaggeration=16,  # early exaggeration alpha 16 is good
                                          learning_rate=max(200, features_10fps_scaled[n].shape[1] / 16),  # alpha*eta = n
                                          **tsne_params).fit_transform(features_10fps_scaled[n].T)
            trained_tsne.append(trained_tsne_i)
            logger.info('Done embedding into 3 D.')
        if comp == 1:  # TODO: review the original implementation...should it be in the loop?
            scaler = StandardScaler()
            scaler.fit(features_10fps.T)
            features_10fps_scaled = scaler.transform(features_10fps.T).T
            logger.info(f'Training t-SNE to embed {features_10fps_scaled.shape[1]} instances from {features_10fps_scaled.shape[0]} D into 3 D from a total of {len(data)} CSV files...')
            trained_tsne = TSNE_sklearn(
                perplexity=np.sqrt(features_10fps_scaled.shape[1]),  # Perplexity scales with sqrt, power law
                early_exaggeration=16,  # early exaggeration alpha 16 is good
                learning_rate=max(200, features_10fps_scaled.shape[1] / 16),  # alpha*eta = n
                **tsne_params
            ).fit_transform(features_10fps_scaled.T)

            logger.debug(f'{inspect.stack()[0][3]}(): Done embedding into 3 D.')
    return features_10fps, features_10fps_scaled, trained_tsne

@config.deco__log_entry_exit(logger)
def train_emgmm_with_learned_tsne_space(trained_tsne_array, comp=config.COMPILE_CSVS_FOR_TRAINING, emgmm_params=config.EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne_array: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logger.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne_array.shape))
        clf_gmm = GaussianMixture(**emgmm_params).fit(trained_tsne_array)
        logger.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne_array.shape))
        assignments = clf_gmm.predict(trained_tsne_array)
    elif comp == 0:
        assignments: List[np.ndarray] = []
        for assignment in tqdm(range(len(trained_tsne_array))):
            logger.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne_array[assignment].shape))
            clf_gmm = GaussianMixture(**emgmm_params).fit(trained_tsne_array[assignment])
            logger.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne_array[assignment].shape))
            assign = clf_gmm.predict(trained_tsne_array[assignment])
            assignments.append(assign)

    logger.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne_array.shape))
    uk = list(np.unique(assignments))
    assignments_list: List[int] = []
    for assignment in assignments:
        assignment_idx_value = uk.index(assignment)
        assignments_list += [assignment_idx_value]
    assignments: np.ndarray = np.array(assignments_list)
    return assignments
def train_em_gmm_NEW(df):
    """"""
    # TODO: HIGH
    return
@config.deco__log_entry_exit(logger)
def bsoid_svm_py(features: np.ndarray, labels: np.ndarray,
                 comp: int = config.COMPILE_CSVS_FOR_TRAINING,
                 holdout_pct: float = config.HOLDOUT_PERCENT,
                 cv_it: int = config.CROSSVALIDATION_K,
                 svm_params: dict = config.SVM_PARAMS) -> Tuple[Any, Any]:
    # TODO: low: depending on COMP value, could return two lists or a classifier and a list...consistency?!!!!
    """
    Train SVM classifier
    :param comp:
    :param features: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param holdout_pct: (float) Test partition ratio for validating SVM performance in GLOBAL_CONFIG
    :param cv_it: (int) iterations for cross-validation in GLOBAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    time_str = config.runtime_timestr  # time_str = time.strftime("_%Y%m%d_%H%M")

    if comp == 1:
        # TODO: ***: You don't need to train/test split if you're using crossfold validation!!!
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            features.T, labels.T, test_size=holdout_pct, random_state=config.RANDOM_STATE)
        logger.info(f'Training SVM on randomly partitioned {(1-holdout_pct)*100}% of training data...')
        classifier = SVC(**svm_params)
        classifier.fit(feats_train, labels_train)
        logger.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
        logger.info(f'Predicting randomly sampled (non-overlapped) assignments '
                    f'using the remaining {holdout_pct * 100}%...')
        scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=config.CROSSVALIDATION_N_JOBS)
        if config.PLOT_GRAPHS:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            title_names, j = ["counts", "norm"], 0
            for title, normalize in titles_options:
                display = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                cmap=plt.cm.Blues, normalize=normalize)
                display.ax_.set_title(title)
                print(title)
                print(display.confusion_matrix)
                if config.SAVE_GRAPHS_TO_FILE:
                    my_file = f'confusion_matrix_{title_names[j]}'
                    file_name = f'{my_file}_{time_str}'
                    visuals.save_graph_to_file(display.figure_, file_name)
                j += 1
            plt.show()
    else:
        classifier, scores = [], []
        for i in range(len(features)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(
                features[i].T, labels[i].T, test_size=holdout_pct, random_state=config.RANDOM_STATE)
            logger.debug(f'{inspect.stack()[0][3]}(): Training SVM on randomly partitioned {(1-holdout_pct)*100}% of training data...')
            clf = SVC(**svm_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logger.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
            logger.info(f'Predicting randomly sampled (non-overlapped) assignments using '
                        f'the remaining {holdout_pct * 100}%...')
            # sc = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=config.crossvalidation_n_jobs)  # TODO: low: `sc` unused variable
            if config.PLOT_GRAPHS:
                np.set_printoptions(precision=2)
                j = 0
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true'), ]
                title_names = ["counts", "norm"]
                for title, normalize in titles_options:
                    display = plot_confusion_matrix(classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
                    display.ax_.set_title(title)
                    print(title)
                    print(display.confusion_matrix)
                    if config.SAVE_GRAPHS_TO_FILE:
                        my_file = f'confusion_matrix_clf{i}_{title_names[j]}_{time_str}'
                        visuals.save_graph_to_file(display.figure_, my_file)
                    j += 1
                plt.show()
    logger.info(f'{inspect.stack()[0][3]}(): Scored cross-validated SVM performance.')  # Previously: .format(feats_train.shape, labels_train.shape))
    return classifier, scores

@config.deco__log_entry_exit(logger)
def get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(train_folders: List[str]):
    """
    This function takes the place of "main.py" previously implemented in bsoid_py.

    :param train_folders: (List[str])
    :return:
    """
    # Check args
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}.')
    if len(train_folders) == 0:
        zero_train_folders_error = f'{inspect.stack()[0][3]}: zero train folders were input for ' \
                                   f'arg `train values` (value = {train_folders}).'
        logger.error(zero_train_folders_error)
        raise ValueError(zero_train_folders_error)

    ##########
    # Get data
    file_names_list, list_of_arrays_of_training_data, _perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)
    # Check that outputs are fine for runtime
    if len(file_names_list) == 0:
        zero_folders_error = f'{inspect.stack()[0][3]}: Zero training folders were specified. Check ' \
                             f'your config file!!! Train folders = {train_folders} // Filenames = {file_names_list}.'
        logger.error(zero_folders_error)
        raise ValueError(zero_folders_error)
    if len(file_names_list[0]) == 0:
        zero_filenames_error = f'{inspect.stack()[0][3]}: Zero file names were found. filenames = {file_names_list}.'
        logger.error(zero_filenames_error)
        raise ValueError(zero_filenames_error)

    # Train TSNE
    features_10fps, features_10fps_scaled, trained_tsne_list, scaler = extract_features_and_train_TSNE(list_of_arrays_of_training_data)  # features_10fps, features_10fps_scaled, trained_tsne_list, scaler = bsoid_tsne_py(list_of_arrays_of_training_data)  # replace with: extract_features_and_train_TSNE

    # Train GMM
    gmm_assignments = train_emgmm_with_learned_tsne_space(trained_tsne_list)  # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below

    # Train SVM
    classifier, scores = bsoid_svm_py(features_10fps_scaled, gmm_assignments)

    # Plot to view progress if necessary
    if config.PLOT_GRAPHS:
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        visuals.plot_GM_assignments_in_3d(trained_tsne_list, gmm_assignments, config.SAVE_GRAPHS_TO_FILE)
        visuals.plot_accuracy_SVM(
            scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')
        visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')
    return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores

@config.deco__log_entry_exit(logger)
def train__import_data_and_process__train_tsne__train_gmm__train_clf__voc(train_folders: list):
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}).')

    _file_names, training_data, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)

    # Train T-SNE
    features_10fps, features_10fps_scaled, trained_tsne = bsoid_tsne_voc(training_data)

    # Train GMM
    # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below
    gmm_assignments = train_emgmm_with_learned_tsne_space(trained_tsne)

    # Train classifier
    classifier, scores = train_mlp_classifier_voc(features_10fps, gmm_assignments)

    # Plot if necessary
    if config.PLOT_GRAPHS:
        visuals.plot_classes_bsoidvoc(trained_tsne, gmm_assignments)
        visuals.plot_accuracy_bsoidvoc(scores)
        visuals.plot_feats_bsoidvoc(features_10fps, gmm_assignments)
    return features_10fps, trained_tsne, gmm_assignments, classifier, scores


### Legacy functions -- keep them for now ########################################################################
def bsoid_tsne_py(list_of_arrays_data: List[np.ndarray], bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING):
    replacement_func = extract_features_and_train_TSNE
    warning = f'This function, {inspect.stack()[0][3]}, will be replaced ' \
              f'with {replacement_func.__qualname__}. Caller = {inspect.stack()[1][3]}.'
    logger.warning(warning)
    return replacement_func(list_of_arrays_data, bodyparts=bodyparts, fps=fps, comp=comp)
def bsoid_nn_voc(feats, labels, comp: int = config.COMPILE_CSVS_FOR_TRAINING, hldout: float = config.HOLDOUT_PERCENT, cv_it=config.CROSSVALIDATION_K, mlp_params=config.MLP_PARAMS):
    # WARNING: DEPRECATION IMMINENT
    replacement_func = train_mlp_classifier_voc
    logger.warn(f'This function will be deprecated in the future. If you still need this function to use, '
                f'think about using {replacement_func.__qualname__} instead. Caller = {inspect.stack()[1][3]}.')
    return replacement_func(feats, labels, comp, hldout, cv_it, mlp_params)
def bsoid_gmm_pyvoc(trained_tsne_array, comp=config.COMPILE_CSVS_FOR_TRAINING, emgmm_params=config.EMGMM_PARAMS) -> np.ndarray:
    replacement_func = train_emgmm_with_learned_tsne_space
    logger.warn(f'This function will be deprecated in the future. To resolve this warning, replace this '
                f'function with {replacement_func.__qualname__} instead.')
    return replacement_func(trained_tsne_array, comp=comp, emgmm_params=emgmm_params)
def bsoid_feats_umapapp(data: list, fps: int = config.VIDEO_FPS) -> Tuple:
    replacement_func = train_umap_unsupervised_with_xy_features_umapapp
    logger.warn(f'DEPRECATION WARNING. This function, {inspect.stack()[0][3]}, will be deprecated in'
                f' favour of a more clear '
                f'and concise function. Caller = {inspect.stack()[1][3]}. '
                f'Current replacement is: {replacement_func.__qualname__}. '
                f'This function only still exists to ensure dependencies aren\'t broken on updating entire module')
    return replacement_func(data, fps)


@config.deco__log_entry_exit(logger)
def main_py(*args, **kwargs):
    """
    :param train_folders: list, training data folders
    :return f_10fps: 2D array, features
    :return trained_tsne: 2D array, trained t-SNE space
    :return gmm_assignments: Converged EM-GMM group assignments
    :return classifier: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    """
    """ *** DEPRECATION WARNING ***
    Only remove after README is updated. """
    replacement_func = get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py
    err = f'Use `{replacement_func.__qualname__}` instead'
    logger.error(err)
    raise DeprecationWarning(err)
@config.deco__log_entry_exit(logger)
def main_voc(train_folders: list):
    replacement_func = train__import_data_and_process__train_tsne__train_gmm__train_clf__voc
    warning = f'This function, {likelihoodprocessing.get_current_function()}, will be deprecated soon. Instead, use: ' \
              f'{replacement_func.__qualname__}.'
    logger.warning(warning)
    return replacement_func(train_folders)
@config.deco__log_entry_exit(logger)
def main_umap(train_folders: list):
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}')
    time_str = config.runtime_timestr  # time_str = time.strftime("_%Y%m%d_%H%M")
    _filenames, training_data, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)
    features_10fps, features_10fps_scaled = train_umap_unsupervised_with_xy_features_umapapp(training_data)

    # Train UMAP (unsupervised) given a set of features based on (x,y) positions
    trained_umap, umap_embeddings = bsoid_umap_embed_umapapp(features_10fps_scaled)
    # Train HDBSCAN (unsupervised) given learned UMAP space
    hdb_assignments, soft_clusters, soft_assignments = bsoid_hdbscan_umapapp(umap_embeddings)
    # Train UMAP
    nn_classifier, scores, nn_assignments = bsoid_nn_appumap(features_10fps, soft_assignments)

    if config.PLOT_GRAPHS:
        fig = visuals.plot_classes_bsoidumap(
            umap_embeddings[hdb_assignments >= 0], hdb_assignments[hdb_assignments >= 0])
        if config.SAVE_GRAPHS_TO_FILE:
            fig_filename = f'HDBSCAN_soft_assignments_{time_str}'  # fig1.savefig(os.path.join(config.OUTPUT_PATH, str.join('', ('hdb_soft_assignments', time_str, '.svg'))))
            visuals.save_graph_to_file(fig, fig_filename)
        visuals.plot_accuracy_bsoidumap(scores)

    return features_10fps, features_10fps_scaled, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments


########################################################################################################################

if __name__ == '__main__':
    get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(config.TRAIN_FOLDERS_IN_DLC_PROJECT_toBeDeprecated)  # originally: main()
    pass
