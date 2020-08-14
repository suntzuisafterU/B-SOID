############### bsoid app###########################################################################
"""
Classify behaviors based on (x,y) using trained B-SOiD behavioral model.
B-SOiD behavioral model has been developed using bsoid_app.main.build()
"""

# # # General imports # # #
from typing import Tuple
import itertools
import logging
import math
import numpy as np

# # # B-SOiD imports # # #
from bsoid.util import likelihoodprocessing, statistics, videoprocessing, visuals
from bsoid.config.LOCAL_CONFIG import BASE_PATH, BODYPARTS, COMP, FPS, FRAME_DIR, ID, GEN_VIDEOS, MODEL_NAME, OUTPUT_PATH, PLOT_TRAINING, VID, VID_NAME
from bsoid.config.GLOBAL_CONFIG import SVM_PARAMS


def bsoid_extract_app(data, fps):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info(f'Extracting features from CSV file {m+1}...')
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
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    logging.info(f'Done extracting features from a total of {len(data)} training CSV files.')
    f_10fps = []
    for n in range(len(feats)):
        feats1 = np.zeros(len(data[n]))
        for s in range(math.floor(fps / 10)):
            for k in range(round(fps / 10) + s, len(feats[n][0]), round(fps / 10)):
                    if k > round(fps / 10) + s:
                        feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                                 np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                                     range(k - round(fps / 10), k)]), axis=1),
                                                            np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                                    range(k - round(fps / 10), k)]),
                                                                   axis=1))).reshape(len(feats[0]), 1)), axis=1)
                    else:
                        feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]),
                                                    axis=1),
                                            np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                    range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
            logging.info('Done integrating features into 100ms bins from CSV file {n + 1}.')
            f_10fps.append(feats1)
    return f_10fps
def bsoid_extract_umap(data, fps=FPS):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logging.info(f'Extracting features from CSV file {m + 1}...')
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
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    f_10fps = []
    for n in range(len(features)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10), len(features[n][0]), round(fps / 10)):
            if k > round(fps / 10):
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
        logging.info(f'Done integrating features into 100ms bins from CSV file {n + 1}.')
        f_10fps.append(feats1)
    return f_10fps
def bsoid_extract_py(data, bodyparts: dict = BODYPARTS, fps: int = FPS):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param bodyparts: dict, body parts with their orders
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logging.info(f'Extracting features from CSV file {m + 1}...')
        data_range = len(data[m])
        fpd = data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1'):2 * bodyparts.get('Forepaw/Shoulder1') + 2] - \
              data[m][:, 2 * bodyparts.get('Forepaw/Shoulder2'):2 * bodyparts.get('Forepaw/Shoulder2') + 2]
        cfp = np.vstack(((data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1')] +
                          data[m][:, 2 * bodyparts.get('Forepaw/Shoulder2')]) / 2,
                         (data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1] +
                          data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1]) / 2)).T
        cfp_pt = np.vstack(([cfp[:, 0] - data[m][:, 2 * bodyparts.get('Tailbase')],
                             cfp[:, 1] - data[m][:, 2 * bodyparts.get('Tailbase') + 1]])).T
        chp = np.vstack((((data[m][:, 2 * bodyparts.get('Hindpaw/Hip1')] +
                           data[m][:, 2 * bodyparts.get('Hindpaw/Hip2')]) / 2),
                         ((data[m][:, 2 * bodyparts.get('Hindpaw/Hip1') + 1] +
                           data[m][:, 2 * bodyparts.get('Hindpaw/Hip2') + 1]) / 2))).T
        chp_pt = np.vstack(([chp[:, 0] - data[m][:, 2 * bodyparts.get('Tailbase')],
                             chp[:, 1] - data[m][:, 2 * bodyparts.get('Tailbase') + 1]])).T
        sn_pt = np.vstack(([data[m][:, 2 * bodyparts.get('Snout/Head')] - data[m][:, 2 * bodyparts.get('Tailbase')],
                            data[m][:, 2 * bodyparts.get('Snout/Head') + 1] - data[m][:,
                                                                              -2 * bodyparts.get('Tailbase') + 1]])).T
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
        for k in range(data_range-1):  # TODO: low: address range starts at 0
            b_3d = np.hstack([sn_pt[k + 1, :], 0])
            a_3d = np.hstack([sn_pt[k, :], 0])
            c = np.cross(b_3d, a_3d)
            sn_pt_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                  math.atan2(np.linalg.norm(c), np.dot(sn_pt[k, :], sn_pt[k + 1, :])))
            sn_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1] -
                data[m][k, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1])
            pt_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1] -
                data[m][k, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1])
        sn_pt_ang_smth = likelihoodprocessing.boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = likelihoodprocessing.boxcar_center(sn_disp, win_len)
        pt_disp_smth = likelihoodprocessing.boxcar_center(pt_disp, win_len)
        features.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:],
                                   sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logging.info(f'Done extracting features from a total of {len(data)} training CSV files.')
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
        logging.info(f'Done integrating features into 100ms bins from CSV file {n + 1}.')
        f_10fps.append(feats1)
    return f_10fps
def bsoid_extract_voc(data, bodyparts=BODYPARTS, fps=FPS):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param bodyparts: dict, body parts with their orders
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info('Extracting features from CSV file {}...'.format(m + 1))
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
        feats.append(np.vstack((p15_norm_smth[1:], p18_norm_smth[1:],
                                p12_ang_smth[:], p14_ang_smth[:], p3_disp_smth[:], p7_disp_smth[:])))
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    f_10fps = []
    for n in range(len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate(
                    (feats1.reshape(feats1.shape[0], feats1.shape[1]),
                    np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                np.sum((feats[n][2:6, range(k - round(fps / 10), k)]), axis=1))).reshape(
                         len(feats[0]), 1)
                     ), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(feats[0]), 1)
        logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
        f_10fps.append(feats1)
    return f_10fps


def bsoid_predict_app(feats, clf) -> list:
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(len(feats)):
        labels = clf.predict(feats[i].T)
        logging.info(f'Done predicting file {i+1} with {feats[i].shape[1]} instances in {feats[i].shape[0]} D space.')
        labels_fslow.append(labels)
    logging.info(f'Done predicting a total of {len(feats)} files.')
    return labels_fslow
def bsoid_predict_py(feats, scaler, model):
    """
    :param feats: list, multiple feats (original feature space)
    :param model: Obj, SVM classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fslow = []
    for i in range(len(feats)):  # TODO: low: address range starts at 0
        logging.info(f'Predicting file {i + 1} with {feats[i].shape[1]} instances using '
                     f'learned classifier: bsoid_{MODEL_NAME}...')
        feats_sc = scaler.transform(feats[i].T).T
        labels = model.predict(feats_sc.T)
        logging.info(f'Done predicting file {i + 1} with {feats[i].shape[1]} instances in {feats[i].shape[0]} D space.')
        labels_fslow.append(labels)
    logging.info(f'Done predicting a total of {len(feats)} files.')
    return labels_fslow
def bsoid_predict_umap(feats, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(len(feats)):
        logging.info('Predicting file {} with {} instances '
                     'using learned classifier: {}{}...'.format(i + 1, feats[i].shape[1], 'bsoid_', MODEL_NAME))
        labels = clf.predict(feats[i].T)
        logging.info('Done predicting file {} with {} instances in {} D space.'.format(i + 1, feats[i].shape[1],
                                                                                       feats[i].shape[0]))
        labels_fslow.append(labels)
    logging.info(f'Done predicting a total of {len(feats)} files.')
    return labels_fslow
def bsoid_predict_voc(feats, model):
    """
    :param feats: list, multiple feats (original feature space)
    :param model: Obj, MLP classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fslow = []
    for i in range(len(feats)):
        logging.info('Predicting file {} with {} instances '
                     'using learned classifier: {}{}...'.format(i + 1, feats[i].shape[1], 'bsoid_', MODEL_NAME))
        labels = model.predict(feats[i].T)
        logging.info('Done predicting file {} with {} instances in {} D space.'.format(i + 1, feats[i].shape[1],
                                                                                       feats[i].shape[0]))
        labels_fslow.append(labels)
    logging.info(f'Done predicting a total of {len(feats)} files.')
    return labels_fslow


def bsoid_frameshift_app(data_new, fps, clf):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param clf: Obj, MLP classifier
    :return fs_labels, 1D array, label/frame
    """
    labels_fs = []
    labels_fs2 = []
    labels_fshigh = []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_app(data_offset)
        labels = bsoid_predict_app(feats_new, clf)
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
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fshigh
def bsoid_frameshift_py(data_new, scaler, fps, model):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param scaler: TODO
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param model: Obj, SVM classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_fs = []
    labels_fshigh = []
    for i in range(len(data_new)):  # TODO: low: address range starts at 0
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_py(data_offset)
        labels = bsoid_predict_py(feats_new, scaler, model)
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
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fshigh
def bsoid_frameshift_umap(data_new, fps: int, clf):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param clf: Obj, MLP classifier
    :return fs_labels, 1D array, label/frame
    """
    labels_fs = []
    labels_fs2 = []
    labels_fshigh = []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_umap(data_offset)
        labels = bsoid_predict_umap(feats_new, clf)
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
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info(f'Done frameshift-predicting a total of {len(data_new)} files.')
    return labels_fshigh
def bsoid_frameshift_voc(data_new, fps, model):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param model: Obj, MLP classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_fs = []
    labels_fs2 = []
    labels_fshigh = []
    for i in range(len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract_voc(data_offset)
        labels = bsoid_predict_voc(feats_new, model)
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
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info('Done frameshift-predicting a total of {} files.'.format(len(data_new)))
    return labels_fshigh


def main_py(predict_folders, scaler, fps, behv_model) -> Tuple:
    """
    TODO: low: purpose
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :param behv_model: object, SVM classifier
    :return Tuple:
        data_new: list, csv data
        feats_new: 2D array, extracted features
        labels_fslow, 1D array, label/100ms
        labels_fshigh, 1D array, label/frame
    """
    filenames, data_new, perc_rect = likelihoodprocessing.main(predict_folders)
    feats_new = bsoid_extract_py(data_new)
    labels_fslow = bsoid_predict_py(feats_new, scaler, behv_model)
    labels_fshigh = bsoid_frameshift_py(data_new, scaler, fps, behv_model)
    if PLOT_TRAINING:
        visuals.plot_feats_bsoidpy(feats_new, labels_fslow)
    if GEN_VIDEOS:
        videoprocessing.main(VID_NAME, labels_fslow[ID], FPS, FRAME_DIR)
    return data_new, feats_new, labels_fslow, labels_fshigh
def main_umap(predict_folders, fps, clf):
    """
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :param clf: object, MLP classifier
    :return data_new: list, csv data
    :return fs_labels, 1D array, label/frame
    """
    filenames, data_new, perc_rect = likelihoodprocessing.main(predict_folders)
    fs_labels = bsoid_frameshift_umap(data_new, fps, clf)
    if VID:
        videoprocessing.main(VID_NAME, fs_labels[ID][0:-1:int(round(FPS / 10))], FPS, FRAME_DIR)
    return data_new, fs_labels
def main_voc(predict_folders, fps, behv_model):
    """
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :behv_model: object, MLP classifier
    :return data_new: list, csv data
    :return feats_new: 2D array, extracted features
    :return labels_fslow, 1D array, label/100ms
    :return labels_fshigh, 1D array, label/frame
    """
    filenames, data_new, perc_rect = likelihoodprocessing.main(predict_folders)
    feats_new = bsoid_extract_voc(data_new)
    labels_fslow = bsoid_predict_voc(feats_new, behv_model)
    labels_fshigh = bsoid_frameshift_voc(data_new, fps, behv_model)
    if PLOT_TRAINING:
        visuals.plot_feats_bsoidvoc(feats_new, labels_fslow)
    if GEN_VIDEOS:
        videoprocessing.main(VID_NAME, labels_fslow[ID], FPS, FRAME_DIR)
    return data_new, feats_new, labels_fslow, labels_fshigh
