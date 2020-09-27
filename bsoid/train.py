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

from bsoid import config, feature_engineering, train_LEGACY
from bsoid.util import check_arg, likelihoodprocessing, visuals

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
        rand_seed=config.RANDOM_STATE)

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
    for n, feature_n in enumerate(features):  # for n in range(len(features)):
        feats1 = np.zeros(len(list_of_arrays_data[n]))
        for k in range(round(fps / 10) - 1, len(feature_n[0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feature_n[0:4, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feature_n[4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)), axis=1)
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
    :param df:
    :param label: (str)
    :param features: 2D array, original feature space, standardized
    :param labels:
    :param comp:
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
        # TODO: ? crossval?
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
    file_names_list, list_of_arrays_of_training_data, _perc_rect = \
        likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)

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

    # Extract features and train TSNE
    features_10fps, features_10fps_scaled, trained_tsne_list, scaler = train_LEGACY.extract_features_and_train_TSNE(list_of_arrays_of_training_data)  # features_10fps, features_10fps_scaled, trained_tsne_list, scaler = bsoid_tsne_py(list_of_arrays_of_training_data)  # replace with: extract_features_and_train_TSNE

    # Train GMM
    gmm_assignments = train_LEGACY.train_emgmm_with_learned_tsne_space(trained_tsne_list)  # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below

    # Train SVM
    classifier, scores = bsoid_svm_py(features_10fps_scaled, gmm_assignments)

    # Plot to view progress if necessary
    if config.PLOT_GRAPHS:
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        visuals.plot_classes_EMGMM_assignments(trained_tsne_list, gmm_assignments, config.SAVE_GRAPHS_TO_FILE)
        visuals.plot_accuracy_SVM(
            scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')
        visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')

    return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores


########################################################################################################################

if __name__ == '__main__':
    get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(config.TRAIN_FOLDERS_IN_DLC_PROJECT)  # originally: main()
    pass
