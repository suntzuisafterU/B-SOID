"""
Classify behaviors based on (x,y) using trained B-SOiD behavioral model.
B-SOiD behavioral model has been developed using bsoid_app.main.build()

'behv_model" is assumed to mean "behavioural model"
"""

# # # General imports # # #
from typing import Any, List, Tuple
import inspect
import itertools
# import logger
import math
import numpy as np
import os

# # # B-SOiD imports # # #
from bsoid import classify_LEGACY, config, feature_engineering
from bsoid.util import check_arg, likelihoodprocessing, videoprocessing, visuals

logger = config.initialize_logger(__name__)

""" bsoid_extract___
Extracts features based on (x,y) positions
:param data: list, csv data
:param fps: scalar, input for camera frame-rate
:return f_10fps: 2D array, extracted features
"""


### Previous implementations
def bsoid_extract_app(data, fps) -> List:
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m, data_m in enumerate(data):  # TODO: refactor this function to use data_m instead of indexing everywhere
        logger.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data[m])
        dxy_r, dis_r = [], []
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
        dis_smth = []
        dxy_eu = np.zeros([data_range, dxy_r.shape[1]])
        ang = np.zeros([data_range - 1, dxy_r.shape[1]])
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(likelihoodprocessing.boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(data_range):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < data_range - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(likelihoodprocessing.boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(likelihoodprocessing.boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        features.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    logger.info(f'Done extracting features from a total of {len(data)} training CSV files.')
    # Next, TODO
    features_10fps = []
    for n in range(len(features)):
        feats1 = np.zeros(len(data[n]))
        for s in range(math.floor(fps / 10)):
            for k in range(round(fps / 10) + s, len(features[n][0]), round(fps / 10)):
                    if k > round(fps / 10) + s:
                        feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                                 np.hstack((np.mean((features[n][0:dxy_smth.shape[0],
                                                                     range(k - round(fps / 10), k)]), axis=1),
                                                            np.sum((features[n][dxy_smth.shape[0]:features[n].shape[0],
                                                                    range(k - round(fps / 10), k)]),
                                                                   axis=1))).reshape(len(features[0]), 1)), axis=1)
                    else:
                        feats1 = np.hstack((np.mean((features[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]),
                                                    axis=1),
                                            np.sum((features[n][dxy_smth.shape[0]:features[n].shape[0],
                                                    range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)
            logger.info(f'Done integrating features into 100ms bins from CSV file {n + 1}.')
            features_10fps.append(feats1)
    return features_10fps


def bsoid_predict_py(features, trained_scaler, clf_SVM) -> List:
    """
    :param features: list, multiple feats (original feature space)
    :param trained_scaler:
    :param clf_SVM: Obj, SVM classifier
    :return labels_fslow: list, label/100ms
    """
    labels_frameshifted_low = []  # Originally: labels_fs_low
    for i, feature_i in enumerate(features):
        logger.info(f'{inspect.stack()[0][3]}(): Predicting file {i+1} with {feature_i.shape[1]} instances using '
                    f'learned classifier: bsoid_{config.MODEL_NAME}...')
        feature_i_scaled = trained_scaler.transform(feature_i.T).T
        labels_i = clf_SVM.predict(feature_i_scaled.T)
        logger.info(f'{inspect.stack()[0][3]}(): Done predicting file {i+1} with {feature_i.shape[1]} instances '
                    f'in {features[i].shape[0]} D space.')
        labels_frameshifted_low.append(labels_i)
    logger.info(f'{inspect.stack()[0][3]}(): Done predicting a total of {len(features)} files.')
    return labels_frameshifted_low


def bsoid_predict_app(features, clf_MLP) -> List:
    """
    :param features: list, multiple feats (original feature space)
    :param clf_MLP: Obj, MLP classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fs_low = []
    for i in range(len(features)):
        labels = clf_MLP.predict(features[i].T)
        logger.info(f'Done predicting file {i+1} with {features[i].shape[1]} instances '
                    f'in {features[i].shape[0]} D space.')
        labels_fs_low.append(labels)
    logger.info(f'Done predicting a total of {len(features)} files.')
    return labels_fs_low


def bsoid_frameshift_py(data_new, scaler, fps: int, clf_SVM) -> List[np.ndarray]:
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param scaler: TODO
    :param fps: scalar, argument specifying camera frame-rate
    :param clf_SVM: Obj, SVM classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_frameshifted, labels_frameshifted_high = [], []  # Originally: labels_fs, labels_fs_high

    for i, data_new_i in enumerate(data_new):
        data_offset_for_data_i = []
        for j in range(fps//10):  # Originally: math.floor(fps / 10)
            data_offset_for_data_i.append(data_new_i[j:, :])
        features_new__data_i = classify_LEGACY.bsoid_extract_py(data_offset_for_data_i)
        labels__data_i: List = classify_LEGACY.bsoid_predict_py(features_new__data_i, scaler, clf_SVM)
        for m in range(len(labels__data_i)):
            labels__data_i[m] = labels__data_i[m][::-1]  # Reverse array
        labels_pad = -1 * np.ones([len(labels__data_i), len(max(labels__data_i, key=lambda x: len(x)))])
        for n, label_n in enumerate(labels__data_i):
            labels_pad[n][0:len(label_n)] = label_n
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n-1][0:n]
        labels_frameshifted.append(labels_pad.astype(int))
    # Create labels_frameshifted_high
    for x, labels_fs_x in enumerate(labels_frameshifted):
        labels_fs2 = []
        for z in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs_x[z])
        labels_frameshifted_high.append(np.array(labels_fs2).flatten('F'))
    logger.debug(f'{likelihoodprocessing.get_current_function()}: '
                 f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_frameshifted_high


@config.deco__log_entry_exit(logger)
def main_py(predict_folders: List[str], scaler, fps, svm_classifier__behavioural_model) -> Tuple[Any, List, List, List]:
    """
    :param predict_folders: list, data folders
    :param scaler:
    :param fps: scalar, camera frame-rate
    :param svm_classifier__behavioural_model: object, SVM classifier
    :return Tuple:
        data_new: list, csv data
        feats_new: 2D array, extracted features
        labels_fslow, 1D array, label/100ms
        labels_fshigh, 1D array, label/frame
    """
    # Import preprocessed data
    filenames, data_new, _perc_rect = likelihoodprocessing.\
        import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(predict_folders)
    # Extract features (without 100ms bin integration)
    # features_new = bsoid_extract_py(data_new)  # Implied 100ms bin integration
    features: List[np.ndarray] = feature_engineering.extract_7_features_bsoid_tsne_py(data_new)
    features = feature_engineering.integrate_features_into_100ms_bins_LEGACY(data_new, features)
    # Predict labels
    labels_frameshift_low: List[np.ndarray] = bsoid_predict_py(features, scaler, svm_classifier__behavioural_model)
    # Create
    labels_frameshift_high: List[np.ndarray] = classify_LEGACY.bsoid_frameshift_py(data_new, scaler, fps, svm_classifier__behavioural_model)

    if config.PLOT_GRAPHS:
        visuals.plot_feats_bsoidpy(features, labels_frameshift_low)

    # TODO: HIGH: Ensure that the labels predicted on predict_folders matches to the video that will be labeled hereafter
    if config.GENERATE_VIDEOS:
        if len(labels_frameshift_low) > 0:
            assert os.path.isfile(config.VIDEO_TO_LABEL_PATH), \
                f'Video path is not resolving. Path is now: {config.VIDEO_TO_LABEL_PATH}'  # Debugging effort
            # 1/2 write frames to disk
            # videoprocessing.write_annotated_frames_to_disk_from_video(
            #     config.VIDEO_TO_LABEL_PATH,
            #     labels_frameshift_low[config.IDENTIFICATION_ORDER],
            # )
            videoprocessing.write_annotated_frames_to_disk_from_video_NEW_multiproc(
                config.VIDEO_TO_LABEL_PATH,
                labels_frameshift_low[config.IDENTIFICATION_ORDER],
            )

            ##################################################################################
            # # 2/2 created labeled video
            videoprocessing.create_labeled_vid(
                labels_frameshift_low[config.IDENTIFICATION_ORDER],
                critical_behaviour_minimum_duration=3,
                num_randomly_generated_examples=5,
                frame_dir=config.FRAMES_OUTPUT_PATH,
                output_path=config.SHORT_VIDEOS_OUTPUT_PATH
            )
            # videoprocessing.create_labeled_vid_NEW_ATTEMPT(
            #     labels=labels_frameshift_low[config.IDENTIFICATION_ORDER]
            # )
        else:
            logger.error(f'{inspect.stack()[0][3]}(): config.GENERATE_VIDEOS = {config.GENERATE_VIDEOS}; '
                         f'however, the generation of '
                         f'a video could NOT occur because labels_fs_low is a list of length zero and '
                         f'config.ID is attempting to index an empty list.')

    return data_new, features, labels_frameshift_low, labels_frameshift_high


### Legacy functions (do not delete yet, do not alter)
def bsoid_extract_py(data, bodyparts: dict = config.BODYPARTS_PY_LEGACY, fps: int = config.VIDEO_FPS) -> List:
    """
    * This is a pull from the original bsoid_py implementation. Do not add modify this function! *

    The two main functions that occur here are:
        1: extract 7 features
        2: integreate features into 100ms bins

    NOTE: this function assumes that user wants features integrated into 100ms bins -- if you want a less
    tightly coupled function or want to pipeline feature transformations overtly, check other implementations.

    :param data:
    :param bodyparts:
    :param fps:
    :return:
    """
    replacement_1 = feature_engineering.extract_7_features_bsoid_tsne_py
    replacement_2 = feature_engineering.integrate_features_into_100ms_bins_LEGACY
    logger.warning(f'This function, {inspect.stack()[0][3]}(), is the old but '
                   f'correct implementation of feature extraction for bsoid_py.'
                   f'Caller = {inspect.stack()[1][3]}. Likely to be replaced by 2 functions that split up the work:'
                   f'replace_1 = {replacement_1.__qualname__} / replace_2 = {replacement_2.__qualname__}.'
                   f'In fact, these functions may end up deprecated because new feature engineering '
                   f'pipelining is being implemented :) Stay tuned!')
    if isinstance(data, np.ndarray):
        data = list(data)

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m, data_array in enumerate(data):
        logger.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data_array)
        fpd = data_array[:, 2 * bodyparts.get('Forepaw/Shoulder1'):2 * bodyparts.get('Forepaw/Shoulder1') + 2] - \
              data_array[:, 2 * bodyparts.get('Forepaw/Shoulder2'):2 * bodyparts.get('Forepaw/Shoulder2') + 2]
        cfp = np.vstack(((data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] +
                          data_array[:, 2 * bodyparts.get('Forepaw/Shoulder2')]) / 2,
                         (data_array[:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1] +
                          data_array[:, 2 * bodyparts.get('Forepaw/Shoulder2') + 1]) / 2)).T
        cfp_pt = np.vstack(([cfp[:, 0] - data_array[:, 2 * bodyparts.get('Tailbase')],
                             cfp[:, 1] - data_array[:, 2 * bodyparts.get('Tailbase') + 1]])).T
        chp = np.vstack((((data_array[:, 2 * bodyparts.get('Hindpaw/Hip1')] +
                           data_array[:, 2 * bodyparts.get('Hindpaw/Hip2')]) / 2),
                         ((data_array[:, 2 * bodyparts.get('Hindpaw/Hip1') + 1] +
                           data_array[:, 2 * bodyparts.get('Hindpaw/Hip2') + 1]) / 2))).T
        chp_pt = np.vstack(([chp[:, 0] - data_array[:, 2 * bodyparts.get('Tailbase')],
                             chp[:, 1] - data_array[:, 2 * bodyparts.get('Tailbase') + 1]])).T
        sn_pt = np.vstack(([data_array[:, 2 * bodyparts.get('Snout/Head')] - data_array[:, 2 * bodyparts.get('Tailbase')],
                            data_array[:, 2 * bodyparts.get('Snout/Head') + 1] - data_array[:, 2 * bodyparts.get('Tailbase') + 1]])).T
        fpd_norm = np.zeros(data_range)
        cfp_pt_norm = np.zeros(data_range)
        chp_pt_norm = np.zeros(data_range)
        sn_pt_norm = np.zeros(data_range)
        for i in range(1, data_range):
            fpd_norm[i] = np.array(np.linalg.norm(fpd[i, :]))
            cfp_pt_norm[i] = np.linalg.norm(cfp_pt[i, :])
            chp_pt_norm[i] = np.linalg.norm(chp_pt[i, :])
            sn_pt_norm[i] = np.linalg.norm(sn_pt[i, :])
        fpd_norm_smth = likelihoodprocessing.boxcar_center(fpd_norm, win_len)
        sn_cfp_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm - cfp_pt_norm, win_len)
        sn_chp_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm - chp_pt_norm, win_len)
        sn_pt_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm, win_len)

        sn_pt_ang = np.zeros(data_range - 1)
        sn_disp = np.zeros(data_range - 1)
        pt_disp = np.zeros(data_range - 1)
        for k in range(data_range-1):
            b_3d = np.hstack([sn_pt[k + 1, :], 0])
            a_3d = np.hstack([sn_pt[k, :], 0])
            c = np.cross(b_3d, a_3d)
            sn_pt_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                  math.atan2(np.linalg.norm(c), np.dot(sn_pt[k, :], sn_pt[k + 1, :])))
            sn_disp[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1] - data_array[k, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1])
            pt_disp[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1] - data_array[k, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1])
        sn_pt_ang_smth = likelihoodprocessing.boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = likelihoodprocessing.boxcar_center(sn_disp, win_len)
        pt_disp_smth = likelihoodprocessing.boxcar_center(pt_disp, win_len)
        features.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:],
                                   sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logger.info(f'{inspect.stack()[0][3]}(): Done extracting features from a total of {len(data)} training CSV files.')

    # Integrating into 100ms bins
    features_10fps = []
    for n in range(len(features)):  # TODO: low: address range starts at 0
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(features[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((features[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((features[n][4:7, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((features[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((features[n][4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(features[0]), 1)
        logger.info(f'{inspect.stack()[0][3]}(): Done integrating features into 100ms bins from CSV file {n + 1}.')
        features_10fps.append(feats1)
    return features_10fps
