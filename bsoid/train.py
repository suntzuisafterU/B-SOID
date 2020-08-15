"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.
"""

from bhtsne import tsne
from sklearn import mixture, svm
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List, Tuple
import hdbscan
import itertools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import umap


from bsoid.config.LOCAL_CONFIG import BODYPARTS, COMP, FRAME_DIR, FPS, FRAME_DIR, GEN_VIDEOS, MODEL_NAME, OUTPUT_PATH, PLOT, PLOT_TRAINING, TRAIN_FOLDERS, PREDICT_FOLDERS, VID_NAME
from bsoid.config.GLOBAL_CONFIG import CV_IT, EMGMM_PARAMS, HDBSCAN_PARAMS, HLDOUT, MLP_PARAMS, UMAP_PARAMS, SVM_PARAMS, TSNE_PARAMS
from bsoid.util.likelihoodprocessing import boxcar_center
from bsoid.util.visuals import plot_accuracy, plot_classes, plot_feats
# from bsoid_umap.utils.visuals import plot_accuracy, plot_classes
import bsoid



def bsoid_feats_umapapp(data: list, fps=FPS) -> Tuple:
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized/session features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info('Extracting features from CSV file {}...'.format(m + 1))
        dataRange = len(data[m])
        dxy_r = []
        dis_r = []
        for r in range(dataRange):
            if r < dataRange - 1:
                dis = []
                for c in range(0, data[m].shape[1], 2):
                    dis.append(np.linalg.norm(data[m][r + 1, c:c + 2] - data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(0, data[m].shape[1], 2), 2):
                dxy.append(data[m][r, i:i + 2] - data[m][r, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)
        dis_smth = []
        dxy_eu = np.zeros([dataRange, dxy_r.shape[1]])
        ang = np.zeros([dataRange - 1, dxy_r.shape[1]])
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(dataRange):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < dataRange - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10), len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10):
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                             range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                            range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                            range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
        logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
        if n > 0:
            f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
        else:
            f_10fps = feats1
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = feats1_sc  # scaling is important as I've seen wildly different stdev/feat between sessions
    return f_10fps, f_10fps_sc


def bsoid_umap_embed_umapapp(f_10fps_sc, umap_params=UMAP_PARAMS):
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param f_10fps_sc: 2D array, standardized/session features
    :param umap_params: dict, UMAP params in GLOBAL_CONFIG
    :return trained_umap: object, trained UMAP transformer
    :return umap_embeddings: 2D array, embedded UMAP space
    """
    ###### So far, use of PCA is not necessary. If, however, features go beyond 100, consider taking top 50 PCs #####
    # if f_10fps_sc.shape[0] > 50:
    #     logging.info('Compressing {} instances from {} D '
    #                  'into {} D using PCA'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
    #                                               50))
    #     feats_train = PCA(n_components=50, random_state=23).fit_transform(f_10fps_sc.T)
    #     pca = PCA(n_components=50).fit(f_10fps_sc.T)
    #     logging.info('Done linear transformation with PCA.')
    #     logging.info('The top {} Principal Components '
    #                  'explained {}% variance'.format(50, 100 * np.sum(pca.explained_variance_ratio_)))
    ################ FastICA potentially useful for demixing signal ################
    # lowd_feats = FastICA(n_components=10, random_state=23).fit_transform(f_10fps.T)
    # feats_train = lowd_feats
    feats_train = f_10fps_sc.T
    logging.info('Transforming all {} instances from {} D into {} D'.format(feats_train.shape[0],
                                                                            feats_train.shape[1],
                                                                            umap_params.get('n_components')))
    trained_umap = umap.UMAP(n_neighbors=int(round(np.sqrt(feats_train.shape[0]))),  # power law
                             **umap_params).fit(feats_train)
    umap_embeddings = trained_umap.embedding_
    logging.info('Done non-linear transformation with UMAP from {} D into {} D.'.format(feats_train.shape[1],
                                                                                        umap_embeddings.shape[1]))
    return trained_umap, umap_embeddings


def bsoid_hdbscan_app(umap_embeddings, hdbscan_params=HDBSCAN_PARAMS):
    """
    Trains HDBSCAN (unsupervised) given learned UMAP space
    :param umap_embeddings: 2D array, embedded UMAP space
    :param hdbscan_params: dict, HDBSCAN params in GLOBAL_CONFIG
    :return assignments: HDBSCAN assignments
    """
    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = range(6, 21)
    logging.info('Running HDBSCAN on {} instances in {} D space...'.format(*umap_embeddings.shape))
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                             min_cluster_size=int(round(0.001 * min_c * umap_embeddings.shape[0])),
                                             **hdbscan_params).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            logging.info('Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*umap_embeddings.shape))
    return assignments, soft_clusters, soft_assignments
def bsoid_hdbscan_umap(umap_embeddings, hdbscan_params=HDBSCAN_PARAMS):
    """
    Trains HDBSCAN (unsupervised) given learned UMAP space
    :param umap_embeddings: 2D array, embedded UMAP space
    :param hdbscan_params: dict, HDBSCAN params in GLOBAL_CONFIG
    :return assignments: HDBSCAN assignments
    """
    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = range(6, 21)
    logging.info('Running HDBSCAN on {} instances in {} D space...'.format(*umap_embeddings.shape))
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                             min_cluster_size=int(round(0.001 * min_c * umap_embeddings.shape[0])),
                                             **hdbscan_params).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            logging.info('Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    # trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
    #                                      min_cluster_size=round(umap_embeddings.shape[0] * 0.007),  # just < 1%/cluster
    #                                      **hdbscan_params).fit(umap_embeddings)
    # assignments = best_clf.labels_
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*umap_embeddings.shape))
    return assignments, soft_clusters, soft_assignments


def bsoid_nn_appumap(feats, labels, hldout=HLDOUT, cv_it=CV_IT, mlp_params=MLP_PARAMS):
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, HDBSCAN assignments
    :param hldout: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return clf: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    :return nn_assignments: 1D array, neural net predictions
    """
    feats_filt = feats[:, labels >= 0]
    labels_filt = labels[labels >= 0]
    feats_train, feats_test, labels_train, labels_test = train_test_split(feats_filt.T, labels_filt.T,
                                                                          test_size=hldout, random_state=23)
    logging.info(
        'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
            (1 - hldout) * 100))
    classifier = MLPClassifier(**mlp_params)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**mlp_params)
    clf.fit(feats_filt.T, labels_filt.T)
    nn_assignments = clf.predict(feats.T)
    logging.info('Done training feedforward neural network '
                 'mapping {} features to {} assignments.'.format(feats_train.shape, labels_train.shape))
    scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
    timestr = time.strftime("_%Y%m%d_%H%M")
    if PLOT:
        np.set_printoptions(precision=2)
        titles_options = [("Non-normalized confusion matrix", None),
                          ("Normalized confusion matrix", 'true')]
        titlenames = [("counts"), ("norm")]
        j = 0
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            my_file = 'confusion_matrix_{}'.format(titlenames[j])
            disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
            j += 1
        plt.show()
    logging.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return clf, scores, nn_assignments
def bsoid_nn_voc(feats, labels, comp=COMP, hldout=HLDOUT, cv_it=CV_IT, mlp_params=MLP_PARAMS):
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param hldout: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return classifier: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    """
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats.T, labels.T, test_size=hldout,
                                                                              random_state=23)
        logging.info(
            'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
                (1 - hldout) * 100))
        classifier = MLPClassifier(**mlp_params)
        classifier.fit(feats_train, labels_train)
        logging.info('Done training feedforward neural network mapping {} features to {} assignments.'.format(
            feats_train.shape, labels_train.shape))
        logging.info('Predicting randomly sampled (non-overlapped) assignments '
                     'using the remaining {}%...'.format(HLDOUT * 100))
        scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
        timestr = time.strftime("_%Y%m%d_%H%M")
        if PLOT_TRAINING:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            titlenames = [("counts"), ("norm")]
            j = 0
            for title, normalize in titles_options:
                disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                             cmap=plt.cm.Blues,
                                             normalize=normalize)
                disp.ax_.set_title(title)
                print(title)
                print(disp.confusion_matrix)
                my_file = 'confusion_matrix_{}'.format(titlenames[j])
                disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
                j += 1
            plt.show()
    else:
        classifier = []
        scores = []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(feats[i].T, labels[i].T,
                                                                                  test_size=hldout,
                                                                                  random_state=23)
            logging.info(
                'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
                    (1 - hldout) * 100))
            clf = MLPClassifier(**mlp_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logging.info(
                'Done training feedforward neural network mapping {} features to {} assignments.'.format(
                    feats_train.shape, labels_train.shape))
            logging.info('Predicting randomly sampled (non-overlapped) assignments '
                         'using the remaining {}%...'.format(HLDOUT * 100))
            sc = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
            timestr = time.strftime("_%Y%m%d_%H%M")
            if PLOT_TRAINING:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                titlenames = [("counts"), ("norm")]
                for title, normalize in titles_options:
                    disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=normalize)
                    disp.ax_.set_title(title)
                    print(title)
                    print(disp.confusion_matrix)
                    my_file = 'confusion_matrix_clf{}_{}'.format(i, titlenames[j])
                    disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
                    j += 1
                plt.show()
    logging.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return classifier, scores


def bsoid_tsne_py(data: list, bodyparts=BODYPARTS, fps=FPS, comp=COMP):
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param bodyparts: dict, body parts with their orders in LOCAL_CONFIG
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info(f'Extracting features from CSV file {m+1}...')
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
                            data[m][:, 2 * bodyparts.get('Snout/Head') + 1] - data[m][:, 2 * bodyparts.get('Tailbase') + 1]])).T
        fpd_norm = np.zeros(data_range)
        cfp_pt_norm = np.zeros(data_range)
        chp_pt_norm = np.zeros(data_range)
        sn_pt_norm = np.zeros(data_range)
        for i in range(1, data_range):
            fpd_norm[i] = np.array(np.linalg.norm(fpd[i, :]))
            cfp_pt_norm[i] = np.linalg.norm(cfp_pt[i, :])
            chp_pt_norm[i] = np.linalg.norm(chp_pt[i, :])
            sn_pt_norm[i] = np.linalg.norm(sn_pt[i, :])
        fpd_norm_smth = boxcar_center(fpd_norm, win_len)
        sn_cfp_norm_smth = boxcar_center(sn_pt_norm - cfp_pt_norm, win_len)
        sn_chp_norm_smth = boxcar_center(sn_pt_norm - chp_pt_norm, win_len)
        sn_pt_norm_smth = boxcar_center(sn_pt_norm, win_len)
        sn_pt_ang = np.zeros(data_range - 1)
        sn_disp = np.zeros(data_range - 1)
        pt_disp = np.zeros(data_range - 1)
        for k in range(0, data_range - 1):
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
        sn_pt_ang_smth = boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = boxcar_center(sn_disp, win_len)
        pt_disp_smth = boxcar_center(pt_disp, win_len)
        feats.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:],
                                sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logging.info(f'Done extracting features from a total of {len(data)} training CSV files.')
    if comp == 0:
        f_10fps = []
        f_10fps_sc = []
        trained_tsne = []
    for n in range(0, len(feats)):  # TODO: low: refactor range
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][4:7, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
        logging.info(f'Done integrating features into 100ms bins from CSV file {n+1}.')
        if comp == 1:
            if n > 0:
                f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            else:
                f_10fps = feats1
        else:
            f_10fps.append(feats1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T).T
            f_10fps_sc.append(feats1_stnd)
            logging.info(f'Training t-SNE to embed {f_10fps_sc[n].shape[1]} instances from '
                         f'{f_10fps_sc[n].shape[0]} D into 3 D from CSV file {n + 1}...')
            trained_tsne_i = tsne(f_10fps_sc[n].T, dimensions=3, perplexity=np.sqrt(f_10fps_sc[n].shape[1]),
                                  theta=0.5, rand_seed=23)
            trained_tsne.append(trained_tsne_i)
            logging.info('Done embedding into 3 D.')
    if comp == 1:
        scaler = StandardScaler()
        scaler.fit(f_10fps.T)  # TODO: HIGH: variable `f_10fps` referenced before assignment. Error in logic above? ########################## IMPORTANT ###############################
        f_10fps_sc = scaler.transform(f_10fps.T).T
        logging.info(f'Training t-SNE to embed {f_10fps_sc.shape[1]} instances from {f_10fps_sc.shape[0]} D '
                     'into 3 D from a total of {len(data)} CSV files...')
        trained_tsne = tsne(f_10fps_sc.T, dimensions=3, perplexity=np.sqrt(f_10fps_sc.shape[1]),
                            theta=0.5, rand_seed=23)  # TODO: low: move "rand_seed" to a config file instead of hiding here as magic variable
        logging.info('Done embedding into 3 D.')
    return f_10fps, f_10fps_sc, trained_tsne, scaler
def bsoid_tsne_voc(data: list, bodyparts=BODYPARTS, fps=FPS, comp=COMP, tsne_params=TSNE_PARAMS):
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
        p15_norm_smth = boxcar_center(p15_norm, win_len)
        p18_norm_smth = boxcar_center(p18_norm, win_len)
        p12_ang = np.zeros(dataRange - 1)
        p14_ang = np.zeros(dataRange - 1)
        p3_disp = np.zeros(dataRange - 1)
        p7_disp = np.zeros(dataRange - 1)
        for k in range(0, dataRange - 1):
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
        p12_ang_smth = boxcar_center(p12_ang, win_len)
        p14_ang_smth = boxcar_center(p14_ang, win_len)
        p3_disp_smth = boxcar_center(p3_disp, win_len)
        p7_disp_smth = boxcar_center(p7_disp, win_len)
        feats.append(np.vstack((p15_norm_smth[1:], p18_norm_smth[1:],
                                p12_ang_smth[:], p14_ang_smth[:], p3_disp_smth[:], p7_disp_smth[:])))
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    if comp == 0:
        f_10fps = []
        f_10fps_sc = []
        trained_tsne = []
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(feats[0]), 1)
        logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
        if comp == 1:
            if n > 0:
                f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            else:
                f_10fps = feats1
        else:
            f_10fps.append(feats1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T).T
            f_10fps_sc.append(feats1_stnd)
            logging.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from CSV file {}...'.format(f_10fps_sc[n].shape[1], f_10fps_sc[n].shape[0],
                                                               n + 1))
            trained_tsne_i = TSNE(perplexity=np.sqrt(f_10fps_sc[n].shape[1]),
                                  early_exaggeration=16,  # early exaggeration alpha 16 is good
                                  learning_rate=max(200, f_10fps_sc[n].shape[1] / 16),  # alpha*eta = n
                                  **tsne_params).fit_transform(f_10fps_sc[n].T)
            trained_tsne.append(trained_tsne_i)
            logging.info('Done embedding into 3 D.')
        if comp == 1:
            scaler = StandardScaler()
            scaler.fit(f_10fps.T)
            f_10fps_sc = scaler.transform(f_10fps.T).T
            logging.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from a total of {} CSV files...'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
                                                                           len(data)))
            trained_tsne = TSNE(perplexity=np.sqrt(f_10fps_sc.shape[1]),  # perplexity scales with sqrt, power law
                                early_exaggeration=16,  # early exaggeration alpha 16 is good
                                learning_rate=max(200, f_10fps_sc.shape[1] / 16),  # alpha*eta = n
                                **tsne_params).fit_transform(f_10fps_sc.T)
            logging.info('Done embedding into 3 D.')
    return f_10fps, f_10fps_sc, trained_tsne


def bsoid_gmm_py(trained_tsne, comp=COMP, emgmm_params=EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne.shape))
        gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne)
        logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
        assigns = gmm.predict(trained_tsne)
    else:
        assigns = []
        for i in tqdm(range(len(trained_tsne))):
            logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne[i].shape))
            gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne[i])
            logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne[i].shape))
            assign = gmm.predict(trained_tsne[i])
            assigns.append(assign)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
    uk = list(np.unique(assigns))
    assignments_li = []
    for i in assigns:
        indexVal = uk.index(i)
        assignments_li.append(indexVal)
    assignments = np.array(assignments_li)
    return assignments
def bsoid_gmm_voc(trained_tsne, comp=COMP, emgmm_params=EMGMM_PARAMS):
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne.shape))
        gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne)
        logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
        assigns = gmm.predict(trained_tsne)
    else:
        assigns = []
        for i in tqdm(range(len(trained_tsne))):
            logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne[i].shape))
            gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne[i])
            logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne[i].shape))
            assign = gmm.predict(trained_tsne[i])
            assigns.append(assign)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
    uk = list(np.unique(assigns))
    assignments_li = []
    for i in assigns:
        indexVal = uk.index(i)
        assignments_li.append(indexVal)
    assignments = np.array(assignments_li)
    return assignments


def bsoid_svm_py(feats, labels, comp=COMP, hldout=HLDOUT, cv_it=CV_IT, svm_params=SVM_PARAMS):
    """
    Trains SVM classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param hldout: scalar, test partition ratio for validating SVM performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            feats.T, labels.T, test_size=hldout, random_state=23)
        logging.info(f'Training SVM on randomly partitioned {(1 - hldout) * 100}% of training data...')
        classifier = svm.SVC(**svm_params)
        classifier.fit(feats_train, labels_train)
        logging.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
        logging.info(f'Predicting randomly sampled (non-overlapped) assignments '
                     f'using the remaining {HLDOUT * 100}%...')
        scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
        timestr = time.strftime("_%Y%m%d_%H%M")
        if PLOT_TRAINING:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            titlenames = [("counts"), ("norm")]  # TODO: low: redundant parentheses
            j = 0
            for title, normalize in titles_options:
                disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                             cmap=plt.cm.Blues, normalize=normalize)
                disp.ax_.set_title(title)
                print(title)
                print(disp.confusion_matrix)
                my_file = f'confusion_matrix_{titlenames[j]}'
                disp.figure_.savefig(os.path.join(OUTPUT_PATH, my_file+timestr+'.svg'))
                j += 1
            plt.show()
    else:
        classifier = []
        scores = []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(
                feats[i].T, labels[i].T, test_size=hldout, random_state=23)
            logging.info(f'Training SVM on randomly partitioned {(1 - hldout) * 100}% of training data...')
            clf = svm.SVC(**svm_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logging.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
            logging.info(f'Predicting randomly sampled (non-overlapped) assignments using '
                         f'the remaining {HLDOUT * 100}%...')
            sc = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)  # TODO: low: `sc` unused variable
            timestr = time.strftime("_%Y%m%d_%H%M")
            if PLOT_TRAINING:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                titlenames = [("counts"), ("norm")]
                for title, normalize in titles_options:
                    disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=normalize)
                    disp.ax_.set_title(title)
                    print(title)
                    print(disp.confusion_matrix)
                    my_file = f'confusion_matrix_clf{i}_{titlenames[j]}'
                    disp.figure_.savefig(os.path.join(OUTPUT_PATH, my_file+timestr+'.svg'))
                    j += 1
                plt.show()
    logging.info('Scored cross-validated SVM performance.'.format(feats_train.shape, labels_train.shape))  # TODO: low: .format() called but variables never used
    return classifier, scores



########################################################################################################################
"""
:param train_folders: list, training data folders
:return f_10fps: 2D array, features
:return trained_tsne: 2D array, trained t-SNE space
:return gmm_assignments: Converged EM-GMM group assignments
:return classifier: obj, MLP classifier
:return scores: 1D array, cross-validated accuracy
"""

def main_py(train_folders: list):
    filenames, training_data, perc_rect = bsoid.util.likelihoodprocessing.main(train_folders)
    f_10fps, f_10fps_sc, trained_tsne, scaler = bsoid_tsne_py(training_data)
    gmm_assignments = bsoid_gmm_py(trained_tsne)
    classifier, scores = bsoid_svm_py(f_10fps_sc, gmm_assignments)
    if PLOT_TRAINING:
        plot_classes(trained_tsne, gmm_assignments)
        plot_accuracy(scores)
        plot_feats(f_10fps, gmm_assignments)
    return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores
def main_umap(train_folders: list):
    filenames, training_data, perc_rect = bsoid.util.likelihoodprocessing.main(train_folders)
    f_10fps, f_10fps_sc = bsoid_feats_umapapp(training_data)
    trained_umap, umap_embeddings = bsoid_umap_embed_umapapp(f_10fps_sc)
    hdb_assignments, soft_clusters, soft_assignments = bsoid_hdbscan_umap(umap_embeddings)
    nn_classifier, scores, nn_assignments = bsoid_nn_appumap(f_10fps, soft_assignments)
    if PLOT:
        timestr = time.strftime("_%Y%m%d_%H%M")
        fig1 = plot_classes(umap_embeddings[hdb_assignments >= 0], hdb_assignments[hdb_assignments >= 0])
        my_file1 = 'hdb_soft_assignments'
        fig1.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file1, timestr, '.svg'))))
        plot_accuracy(scores)
    return f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, \
           nn_classifier, scores, nn_assignments
def main_voc(train_folders: list):
    filenames, training_data, perc_rect = bsoid.util.likelihoodprocessing.main(train_folders)
    f_10fps, f_10fps_sc, trained_tsne = bsoid_tsne_voc(training_data)
    gmm_assignments = bsoid_gmm_voc(trained_tsne)
    classifier, scores = bsoid_nn_voc(f_10fps, gmm_assignments)
    if PLOT_TRAINING:
        plot_classes_voc(trained_tsne, gmm_assignments)
        plot_accuracy_voc(scores)
        plot_feats_voc(f_10fps, gmm_assignments)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores

