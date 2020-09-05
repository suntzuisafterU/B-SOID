"""
Classify behaviors based on (x,y) using trained B-SOiD behavioral model.
B-SOiD behavioral model has been developed using bsoid_app.main.build()
"""

# # # General imports # # #
from typing import Any, List, Tuple
import inspect
import itertools
# import logger
import math
import numpy as np

# # # B-SOiD imports # # #
from bsoid import config
from bsoid.util import check_arg, likelihoodprocessing, videoprocessing, visuals

logger = config.initialize_logger(__name__)

""" bsoid_extract_ _
Extracts features based on (x,y) positions
:param data: list, csv data
:param fps: scalar, input for camera frame-rate
:return f_10fps: 2D array, extracted features
"""


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
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
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
def bsoid_extract_umap(data, fps=config.VIDEO_FPS) -> List:
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logger.debug(f'Extracting features from CSV file {m+1}...')
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
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(likelihoodprocessing.boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(likelihoodprocessing.boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        features.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    logger.info(f'Done extracting features from a total of {len(data)} training CSV files.')

    f_10fps = []
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
        logger.info(f'Done integrating features into 100ms bins from CSV file {n+1}.')
        f_10fps.append(features_n)
    return f_10fps
def extract_7_features_bsoid_tsne_py(list_of_arrays_data: List[np.ndarray], bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING):
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
    # Sometimes data is (incorrectly) submitted as an array of arrays (the number of arrays in the overarching array or, if correctly typed, list) is the same # of CSV files read in). Fix type then continue.
    if isinstance(list_of_arrays_data, np.ndarray):
        list_of_arrays_data = list(list_of_arrays_data)
    # Check args
    check_arg.ensure_type(list_of_arrays_data, list)
    check_arg.ensure_type(list_of_arrays_data[0], np.ndarray)
    # Continue
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    # Iterate over data arrays available and build features
    for i, data_array in enumerate(list_of_arrays_data):  # for i in range(len(list_of_arrays_data)):
        logger.info(f'Extracting features from CSV file {i + 1}...')
        num_data_rows = len(data_array)

        fpd = data_array[:, 2 * bodyparts['Forepaw/Shoulder1']:2 * bodyparts['Forepaw/Shoulder1'] + 2] - data_array[:, 2 * bodyparts['Forepaw/Shoulder2']:2 * bodyparts[ 'Forepaw/Shoulder2'] + 2]
        cfp = np.vstack(((data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2']]) / 2, (data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1] + data_array[:, 2 * bodyparts[ 'Forepaw/Shoulder1'] + 1]) / 2)).T
        cfp_pt = np.vstack(([cfp[:, 0] - data_array[:, 2 * bodyparts['Tailbase']], cfp[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1]])).T
        chp = np.vstack((((data_array[:, 2 * bodyparts['Hindpaw/Hip1']] + data_array[:, 2 * bodyparts['Hindpaw/Hip2']]) / 2), (( data_array[ :, 2 * bodyparts[ 'Hindpaw/Hip1'] + 1] + data_array[ :, 2 * bodyparts[ 'Hindpaw/Hip2'] + 1]) / 2))).T
        chp_pt = np.vstack(([chp[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
                             chp[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1]])).T
        sn_pt = np.vstack(([data_array[:, 2 * bodyparts['Snout/Head']] - data_array[:, 2 * bodyparts['Tailbase']],
                            data_array[:, 2 * bodyparts['Snout/Head'] + 1] - data_array[:,
                                                                             2 * bodyparts['Tailbase'] + 1]])).T

        fpd_norm = np.zeros(num_data_rows)
        cfp_pt_norm = np.zeros(num_data_rows)
        chp_pt_norm = np.zeros(num_data_rows)
        sn_pt_norm = np.zeros(num_data_rows)
        for j in range(1, num_data_rows):
            fpd_norm[j] = np.array(np.linalg.norm(fpd[j, :]))
            cfp_pt_norm[j] = np.linalg.norm(cfp_pt[j, :])
            chp_pt_norm[j] = np.linalg.norm(chp_pt[j, :])
            sn_pt_norm[j] = np.linalg.norm(sn_pt[j, :])

        fpd_norm_smth = likelihoodprocessing.boxcar_center(fpd_norm, win_len)
        sn_cfp_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm - cfp_pt_norm, win_len)
        sn_chp_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm - chp_pt_norm, win_len)
        sn_pt_norm_smth = likelihoodprocessing.boxcar_center(sn_pt_norm, win_len)

        sn_pt_ang = np.zeros(num_data_rows - 1)
        sn_disp = np.zeros(num_data_rows - 1)
        pt_disp = np.zeros(num_data_rows - 1)
        for k in range(num_data_rows - 1):
            b_3d = np.hstack([sn_pt[k + 1, :], 0])
            a_3d = np.hstack([sn_pt[k, :], 0])
            c = np.cross(b_3d, a_3d)
            sn_pt_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                  math.atan2(np.linalg.norm(c), np.dot(sn_pt[k, :], sn_pt[k + 1, :])))
            sn_disp[k] = np.linalg.norm(
                data_array[k + 1, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1] - data_array[k,
                                                                                                 2 * bodyparts[
                                                                                                     'Snout/Head']:2 *
                                                                                                                   bodyparts[
                                                                                                                       'Snout/Head'] + 1])
            pt_disp[k] = np.linalg.norm(
                data_array[k + 1, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1] - data_array[k,
                                                                                             2 * bodyparts[
                                                                                                 'Tailbase']:2 *
                                                                                                             bodyparts[
                                                                                                                 'Tailbase'] + 1])
        sn_pt_ang_smth = likelihoodprocessing.boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = likelihoodprocessing.boxcar_center(sn_disp, win_len)
        pt_disp_smth = likelihoodprocessing.boxcar_center(pt_disp, win_len)

        # Append data to features list
        features.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:], sn_pt_norm_smth[1:],
                                   sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logger.info(
        f'{inspect.stack()[0][3]}:Done extracting features from a total of {len(list_of_arrays_data)} training CSV files.')

    return features
def integrate_features_into_100ms_bins(data: List[np.ndarray], features: List[np.ndarray], fps) -> List[np.ndarray]:
    f_10fps = []
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
        logger.info(f'Done integrating features into 100ms bins from CSV file {n + 1}.')
        f_10fps.append(feats1)
    return f_10fps
def bsoid_extract_features_without_assuming_100ms_bin_integration(data, bodyparts: dict = config.BODYPARTS_PY_LEGACY, fps: int = config.VIDEO_FPS) -> List:
    """
    Originally copied from `bsoid_extract_py()`, this function removed the 100ms bin extraction from the end and
    is now an optional function which can be called separately.
    :param data: TODO: HIGH: List of arrays? or is it an array alone? CHECK!!!!!!!
    :param bodyparts:
    :param fps:
    :return:
    """
    if isinstance(data, np.ndarray):
        data = list(data)

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m, data_array in enumerate(data):
        logger.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data_array)
        fpd = data_array[:, 2 * bodyparts['Forepaw/Shoulder1']:2 * bodyparts['Forepaw/Shoulder1'] + 2] - data_array[:, 2 * bodyparts['Forepaw/Shoulder2']:2 * bodyparts['Forepaw/Shoulder2'] + 2]
        cfp = np.vstack(((data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2']]) / 2, (data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1] + data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1]) / 2)).T
        cfp_pt = np.vstack(([cfp[:, 0] - data_array[:, 2 * bodyparts['Tailbase']], cfp[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1]])).T
        chp = np.vstack((((data_array[:, 2 * bodyparts['Hindpaw/Hip1']] + data_array[:, 2 * bodyparts['Hindpaw/Hip2']]) / 2), ((data_array[:, 2 * bodyparts['Hindpaw/Hip1'] + 1] + data_array[:, 2 * bodyparts['Hindpaw/Hip2'] + 1]) / 2))).T
        chp_pt = np.vstack(([chp[:, 0] - data_array[:, 2 * bodyparts['Tailbase']], chp[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1]])).T
        sn_pt = np.vstack(([data_array[:, 2 * bodyparts['Snout/Head']] - data_array[:, 2 * bodyparts['Tailbase']], data_array[:, 2 * bodyparts['Snout/Head'] + 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1]])).T

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
            sn_disp[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1] - data_array[k, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1])
            pt_disp[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1] - data_array[k, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1])
        sn_pt_ang_smth = likelihoodprocessing.boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = likelihoodprocessing.boxcar_center(sn_disp, win_len)
        pt_disp_smth = likelihoodprocessing.boxcar_center(pt_disp, win_len)
        features.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:],
                                   sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logger.debug(f'Done extracting features from a total of {len(data)} training CSV files.')
    return features
def bsoid_extract_py(data, bodyparts: dict = config.BODYPARTS_PY_LEGACY, fps: int = config.VIDEO_FPS) -> List:
    """
    This is a pull from the original bsoid_py implementation.

        * NOTE: this function assumes that user wants features integrated into 100ms bins -- if you want a less
    tightly coupled function or want to pipeline feature transformations overtly, check other implementations.

    :param data: TODO: HIGH: List of arrays? or is it an array alone? CHECK!!!!!!!
    :param bodyparts:
    :param fps:
    :return:
    """
    if isinstance(data, np.ndarray):
        data = list(data)

    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m, data_array in enumerate(data):
        logger.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data_array)
        fpd = data_array[:, 2 * bodyparts.get('Forepaw/Shoulder1'):2 * bodyparts.get('Forepaw/Shoulder1') + 2] - data_array[:, 2 * bodyparts.get('Forepaw/Shoulder2'):2 * bodyparts.get('Forepaw/Shoulder2') + 2]
        cfp = np.vstack(((data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] +
                          data_array[:, 2 * bodyparts.get('Forepaw/Shoulder2')]) / 2,
                         (data_array[:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1] +
                          data_array[:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1]) / 2)).T
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
    logger.info(f'Done extracting features from a total of {len(data)} training CSV files.')

    f_10fps = []
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
        logger.info(f'Done integrating features into 100ms bins from CSV file {n + 1}.')
        f_10fps.append(feats1)
    return f_10fps
def bsoid_extract_voc(data, bodyparts: dict = config.BODYPARTS_VOC_LEGACY, fps: int = config.VIDEO_FPS) -> List:
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logger.info(f'Extracting features from CSV file {m+1}...')
        dataRange = len(data[m])
        p12 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point2'):2 * bodyparts.get('Point2') + 2]
        p14 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point4'):2 * bodyparts.get('Point4') + 2]
        p15 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point5'):2 * bodyparts.get('Point5') + 2]
        p18 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point8'):2 * bodyparts.get('Point8') + 2]
        p15_norm = np.zeros(dataRange)
        p18_norm = np.zeros(dataRange)
        for i in range(1, dataRange):
            p15_norm[i] = np.array(np.linalg.norm(p15[i, :]))
            p18_norm[i] = np.array(np.linalg.norm(p18[i, :]))
        p15_norm_smth = likelihoodprocessing.boxcar_center(p15_norm, win_len)
        p18_norm_smth = likelihoodprocessing.boxcar_center(p18_norm, win_len)
        p12_ang = np.zeros(dataRange - 1)
        p14_ang = np.zeros(dataRange - 1)
        p3_disp = np.zeros(dataRange - 1)
        p7_disp = np.zeros(dataRange - 1)
        for k in range(dataRange - 1):
            b_3d = np.hstack([p12[k + 1, :], 0])
            a_3d = np.hstack([p12[k, :], 0])
            c = np.cross(b_3d, a_3d)
            p12_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                math.atan2(np.linalg.norm(c), np.dot(p12[k, :], p12[k + 1, :])))
            e_3d = np.hstack([p14[k + 1, :], 0])
            d_3d = np.hstack([p14[k, :], 0])
            f = np.cross(e_3d, d_3d)
            p14_ang[k] = np.dot(np.dot(np.sign(f[2]), 180) / np.pi,
                                math.atan2(np.linalg.norm(f), np.dot(p14[k, :], p14[k + 1, :])))
            p3_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Point3'):2 * bodyparts.get('Point3') + 1] -
                data[m][k, 2 * bodyparts.get('Point3'):2 * bodyparts.get('Point3') + 1])
            p7_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Point7'):2 * bodyparts.get('Point7') + 1] -
                data[m][k, 2 * bodyparts.get('Point7'):2 * bodyparts.get('Point7') + 1])
        p12_ang_smth = likelihoodprocessing.boxcar_center(p12_ang, win_len)
        p14_ang_smth = likelihoodprocessing.boxcar_center(p14_ang, win_len)
        p3_disp_smth = likelihoodprocessing.boxcar_center(p3_disp, win_len)
        p7_disp_smth = likelihoodprocessing.boxcar_center(p7_disp, win_len)
        features.append(np.vstack((p15_norm_smth[1:], p18_norm_smth[1:],
                                p12_ang_smth[:], p14_ang_smth[:], p3_disp_smth[:], p7_disp_smth[:])))
    logger.info(f'Done extracting features from a total of {len(data)} training CSV files.')
    f_10fps = []
    for n in range(len(features)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(features[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((features[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((features[n][2:6, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((features[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((features[n][2:6, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(features[0]), 1)
        logger.info(f'Done integrating features into 100ms bins from CSV file {n+1}.')
        f_10fps.append(feats1)
    return f_10fps


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


def bsoid_predict_py(features, scaler, clf_SVM) -> list:
    """
    :param features: list, multiple feats (original feature space)
    :param scaler:
    :param clf_SVM: Obj, SVM classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fs_low = []
    for i in range(len(features)):
        logger.info(f'Predicting file {i+1} with {features[i].shape[1]} instances using '
                    f'learned classifier: bsoid_{config.MODEL_NAME}...')
        feats_sc = scaler.transform(features[i].T).T
        labels = clf_SVM.predict(feats_sc.T)
        logger.info(f'Done predicting file {i+1} with {features[i].shape[1]} instances '
                    f'in {features[i].shape[0]} D space.')
        labels_fs_low.append(labels)
    logger.info(f'Done predicting a total of {len(features)} files.')
    return labels_fs_low
def bsoid_predict_umapvoc(features, clf_MLP) -> list:
    """ _umapvoc
    :param features: list, multiple feats (original feature space)
    :param clf_MLP: Obj, MLP classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fs_low = []
    for i in range(len(features)):
        logger.info(f'Predicting file {i+1} with {features[i].shape[1]} instances '
                    f'using learned classifier: bsoid_{config.MODEL_NAME}...')
        labels = clf_MLP.predict(features[i].T)
        logger.info(f'Done predicting file {i+1} with {features[i].shape[1]} instances '
                    f'in {features[i].shape[0]} D space.')
        labels_fs_low.append(labels)
    logger.info(f'Done predicting a total of {len(features)} files.')
    return labels_fs_low


def bsoid_frameshift_app(data_new, video_fps: int, clf_MLP) -> List:
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param video_fps: scalar, argument specifying camera frame-rate
    :param clf_MLP: Obj, MLP classifier
    :return fs_labels, 1D array, label/frame
    """
    labels_fs, labels_fs_high = [], []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(video_fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_app(data_offset, video_fps)
        labels = bsoid_predict_app(feats_new, clf_MLP)
        for m in range(len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(video_fps / 10)):
            labels_fs2.append(labels_fs[k][l])
        labels_fs_high.append(np.array(labels_fs2).flatten('F'))
    logger.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fs_high
def bsoid_frameshift_py(data_new, scaler, fps: int, clf_SVM) -> List:
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param scaler: TODO
    :param fps: scalar, argument specifying camera frame-rate
    :param clf_SVM: Obj, SVM classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_fs, labels_fs_high = [], []
    for i in range(len(data_new)):  # TODO: low: address range starts at 0
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_py(data_offset)
        labels = bsoid_predict_py(feats_new, scaler, clf_SVM)
        for m in range(len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(len(labels_fs)):
        labels_fs2 = []
        for z in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs[k][z])
        labels_fs_high.append(np.array(labels_fs2).flatten('F'))
    logger.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fs_high
def bsoid_frameshift_umap(data_new, fps: int, clf_MLP) -> List:
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate
    :param clf_MLP: Obj, MLP classifier
    :return fs_labels, 1D array, label/frame
    """
    labels_fs, labels_fs_high = [], []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_umap(data_offset)
        labels = bsoid_predict_umapvoc(feats_new, clf_MLP)
        for m in range(len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs[k][l])
        labels_fs_high.append(np.array(labels_fs2).flatten('F'))
    logger.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fs_high
def bsoid_frameshift_voc(data_new, fps: int, clf_MLP) -> List:
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate
    :param clf_MLP: Obj, MLP classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_fs, labels_fs_high = [], []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_voc(data_offset)
        labels = bsoid_predict_umapvoc(feats_new, clf_MLP)
        for m in range(len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs[k][l])
        labels_fs_high.append(np.array(labels_fs2).flatten('F'))
    logger.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fs_high

@config.cfig_log_entry_exit(logger)
def main_py(predict_folders: List[str], scaler, fps, behv_model) -> Tuple[np.ndarray, List, List, List]:
    """
    :param predict_folders: list, data folders
    :param scaler:
    :param fps: scalar, camera frame-rate
    :param behv_model: object, SVM classifier
    :return Tuple:
        data_new: list, csv data
        feats_new: 2D array, extracted features
        labels_fslow, 1D array, label/100ms
        labels_fshigh, 1D array, label/frame
    """
    filenames, data_new, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(predict_folders)
    features_new = bsoid_extract_py(data_new)
    labels_fs_low: List = bsoid_predict_py(features_new, scaler, behv_model)
    labels_fs_high: List = bsoid_frameshift_py(data_new, scaler, fps, behv_model)

    if config.PLOT_GRAPHS:
        visuals.plot_feats_bsoidpy(features_new, labels_fs_low)

    if config.GENERATE_VIDEOS:
        if len(labels_fs_low) > 0:
            videoprocessing.get_frames_from_video_then_create_labeled_video(
                path_to_video=config.VIDEO_TO_LABEL_PATH,
                labels=labels_fs_low[config.IDENTIFICATION_ORDER],
                fps=fps,
                output_path=config.FRAME_DIR)
        else:
            logger.error(f'{__name__}::{inspect.stack()[0][3]}::config.GENERATE_VIDEOS = {config.GENERATE_VIDEOS}; '
                         f'however, the generation of '
                         f'a video could NOT occur because labels_fs_low is a list of length zero and '
                         f'config.ID is attempting to index an empty list.')

    return data_new, features_new, labels_fs_low, labels_fs_high
def main_umap(predict_folders: List[str], fps, clf) -> Tuple[np.ndarray, List]:
    """
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :param clf: object, MLP classifier
    :return data_new: list, csv data
    :return fs_labels, 1D array, label/frame
    """

    filenames, data_new, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(predict_folders)

    labels_fs: List = bsoid_frameshift_umap(data_new, fps, clf)

    # if config.GENERATE_VIDEOS:
    #     videoprocessing.get_frames_from_video_then_create_labeled_video(config.VIDEO_TO_LABEL_PATH, labels_fs[config.ID][0:-1:int(round(fps / 10))], fps, config.FRAME_DIR)
    if config.GENERATE_VIDEOS:
        videoprocessing.get_frames_from_video_then_create_labeled_video(
            path_to_video=config.VIDEO_TO_LABEL_PATH,
            labels=labels_fs[config.IDENTIFICATION_ORDER][:-1:int(round(fps / 10))],
            fps=fps,
            output_path=config.FRAME_DIR)

    return data_new, labels_fs
def main_voc(predict_folders: List[str], fps, behv_model) -> Tuple[np.ndarray, Any, Any, Any]:
    """
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :behv_model: object, MLP classifier
    :return data_new: list, csv data
    :return feats_new: 2D array, extracted features
    :return labels_fslow, 1D array, label/100ms
    :return labels_fshigh, 1D array, label/frame
    """

    filenames, data_new, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(predict_folders)

    features_new = bsoid_extract_voc(data_new)
    labels_fs_low = bsoid_predict_umapvoc(features_new, behv_model)
    labels_fs_high = bsoid_frameshift_voc(data_new, fps, behv_model)

    if config.PLOT_GRAPHS:
        visuals.plot_feats_bsoidvoc(features_new, labels_fs_low)
    if config.GENERATE_VIDEOS:
        videoprocessing.get_frames_from_video_then_create_labeled_video(
            path_to_video=config.VIDEO_TO_LABEL_PATH,
            labels=labels_fs_low[config.IDENTIFICATION_ORDER],
            fps=fps,
            output_path=config.FRAME_DIR)

    return data_new, features_new, labels_fs_low, labels_fs_high
