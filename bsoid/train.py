"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.

Potential abbreviations:
    sn: snout
    pt: proximal tail
"""

from bhtsne import tsne
from sklearn import mixture
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, Tuple
import hdbscan
import inspect
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import umap
import warnings

from bsoid import config
from bsoid.util import likelihoodprocessing, visuals

logger = config.create_file_specific_logger(__name__)


########################################################################################################################

def train_umap_unsupervised_with_xy_features_umapapp(data: list, fps: int = config.video_fps) -> Tuple:
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :return (tuple):
        f_10fps: 2D array, features
        f_10fps_sc: 2D array, standardized/session features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features = []
    for m in range(len(data)):
        logger.info(f'Extracting features from CSV file {m + 1}...')
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
                feats1 = np.hstack((np.mean((features[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((features[n][dxy_smth.shape[0]:features[n].shape[0],
                                            range(k - round(fps / 10), k)]), axis=1))).reshape(len(features[0]), 1)
        logger.info(f'{inspect.stack()[0][3]}::Done integrating features into 100ms bins from CSV file {n+1}.')
        if n > 0:  # For any index value of n that isn't the very first run
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


def bsoid_umap_embed_umapapp(f_10fps_sc, umap_params=config.UMAP_PARAMS):
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param f_10fps_sc: 2D array, standardized/session features
    :param umap_params: dict, UMAP params in GLOBAL_CONFIG
    :return trained_umap: object, trained UMAP transformer
    :return umap_embeddings: 2D array, embedded UMAP space
    """
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
    features_train = f_10fps_sc.T
    logger.info(f'Transforming all {features_train.shape[0]} instances '
                f'from {features_train.shape[1]} D into {umap_params.get("n_components")} D')
    trained_umap = umap.UMAP(n_neighbors=int(round(np.sqrt(features_train.shape[0]))),  # power law
                             **umap_params).fit(features_train)
    umap_embeddings = trained_umap.embedding_
    logger.info(f'Done non-linear transformation with UMAP from {features_train.shape[1]} D '
                f'into {umap_embeddings.shape[1]} D.')
    return trained_umap, umap_embeddings


def bsoid_hdbscan_umapapp(umap_embeddings, hdbscan_params=config.HDBSCAN_PARAMS):
    """
    Trains HDBSCAN (unsupervised) given learned UMAP space
    :param umap_embeddings: 2D array, embedded UMAP space
    :param hdbscan_params: dict, HDBSCAN params in GLOBAL_CONFIG
    :return assignments: HDBSCAN assignments
    """
    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = range(6, 21)  # TODO: Q: why is the range this way? Magic variables?
    logger.info('Running HDBSCAN on {} instances in {} D space...'.format(*umap_embeddings.shape))
    for min_c in min_cluster_range:
        # trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
        #                                      min_cluster_size=round(umap_embeddings.shape[0] * 0.007),  # just < 1%/cluster
        #                                      **hdbscan_params).fit(umap_embeddings)
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                             min_cluster_size=int(round(0.001 * min_c * umap_embeddings.shape[0])),
                                             **hdbscan_params)\
                                            .fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            logger.info(f'Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    logger.info('Done predicting labels for {} instances in {} D space...'.format(*umap_embeddings.shape))
    return assignments, soft_clusters, soft_assignments


def bsoid_nn_appumap(feats, labels, holdout_pct: float = config.holdout_percent, cv_it: int = config.crossvalidation_k, mlp_params=config.MLP_PARAMS):
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
    feats_filt = feats[:, labels >= 0]
    labels_filt = labels[labels >= 0]
    feats_train, feats_test, labels_train, labels_test = train_test_split(
        feats_filt.T, labels_filt.T, test_size=holdout_pct, random_state=23)
    logger.info(f'Training feedforward neural network on randomly '
                             f'partitioned {(1 - holdout_pct) * 100}% of training data...')
    classifier = MLPClassifier(**mlp_params)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**mlp_params)
    clf.fit(feats_filt.T, labels_filt.T)
    nn_assignments = clf.predict(feats.T)
    logger.info(f'Done training feedforward neural network mapping {feats_train.shape} features '
                             f'to {labels_train.shape} assignments.')
    scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
    time_str = time.strftime("_%Y%m%d_%H%M")
    if config.PLOT_GRAPHS:
        np.set_printoptions(precision=2)
        titles_options = [("Non-normalized confusion matrix", None),
                          ("Normalized confusion matrix", 'true')]
        titlenames = ["counts", "norm"]
        j = 0
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(
                classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            my_file = f'confusion_matrix_{titlenames[j]}'
            disp.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}{time_str}.svg'))
            j += 1
        plt.show()
    logger.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return clf, scores, nn_assignments


def train_mlp_classifier_voc(feats, labels,
                             comp: int = config.compile_CSVs_for_training,
                             holdout_percent: float = config.holdout_percent,
                             crossvalidation_k:int = config.crossvalidation_k,
                             mlp_params=config.MLP_PARAMS) -> Tuple[Any, Any]:
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
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats.T, labels.T, test_size=holdout_percent,
                                                                              random_state=23)
        logger.info(f'Training feedforward neural network on randomly '
                     f'partitioned {(1 - holdout_percent) * 100}% of training data...')
        classifier = MLPClassifier(**mlp_params)
        classifier.fit(feats_train, labels_train)
        logger.info(f'Done training feedforward neural network mapping {feats_train.shape} features to {labels_train.shape} assignments.')
        logger.info('Predicting randomly sampled (non-overlapped) assignments '
                                 f'using the remaining {holdout_percent * 100}%...')
        scores = cross_val_score(classifier, feats_test, labels_test, cv=crossvalidation_k, n_jobs=-1)
        time_str = time.strftime("_%Y%m%d_%H%M")
        if config.PLOT_TRAINING:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            titlenames = ["counts", "norm"]
            j = 0
            for title, normalize in titles_options:
                display = plot_confusion_matrix(
                    classifier, feats_test, labels_test, cmap=plt.cm.Blues, normalize=normalize)
                display.ax_.set_title(title)
                print(title)
                print(display.confusion_matrix)
                file_name = f'confusion_matrix_{titlenames[j]}'
                display.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{file_name}{time_str}.svg'))
                j += 1
            plt.show()
    else:
        classifier = []
        scores = []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(feats[i].T, labels[i].T,
                                                                                  test_size=holdout_percent,
                                                                                  random_state=23)
            logger.info(
                'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
                    (1 - holdout_percent) * 100))
            clf = MLPClassifier(**mlp_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logger.info(
                'Done training feedforward neural network mapping {} features to {} assignments.'.format(
                    feats_train.shape, labels_train.shape))
            logger.info('Predicting randomly sampled (non-overlapped) assignments '
                         'using the remaining {}%...'.format(holdout_percent * 100))
            sc = cross_val_score(classifier, feats_test, labels_test, cv=crossvalidation_k, n_jobs=-1)
            time_str = time.strftime("_%Y%m%d_%H%M")
            if config.PLOT_TRAINING:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                titlenames = ["counts", "norm"]
                for title, normalize in titles_options:
                    display = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=normalize)
                    display.ax_.set_title(title)
                    print(title)
                    print(display.confusion_matrix)
                    file_name = 'confusion_matrix_clf{}_{}'.format(i, titlenames[j])
                    display.figure_.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (file_name, time_str, '.svg'))))
                    j += 1
                plt.show()
    logger.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return classifier, scores


def bsoid_tsne_py(data: list, bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.video_fps, comp: int = config.compile_CSVs_for_training):
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
    features = []
    for m in range(len(data)):
        logger.info(f'Extracting features from CSV file {m+1}...')
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
                                                                              2 * bodyparts.get('Tailbase') + 1]])).T
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
        for k in range(data_range - 1):
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
    logger.info(f'{inspect.stack()[0][3]}:Done extracting features from a total of {len(data)} training CSV files.')
    if comp == 0:
        f_10fps = []
        f_10fps_sc = []
        trained_tsne = []
    for n in range(len(features)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(features[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((features[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((features[n][4:7, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((features[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((features[n][4:7, range(k - round(fps / 10), k)]), axis=1)))\
                                    .reshape(len(features[0]), 1)
        logger.info(f'Done integrating features into 100ms bins from CSV file {n+1}.')
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
            logger.info(f'Training t-SNE to embed {f_10fps_sc[n].shape[1]} instances from '
                                     f'{f_10fps_sc[n].shape[0]} D into 3 D from CSV file {n + 1}...')
            trained_tsne_i = tsne(f_10fps_sc[n].T, dimensions=3, perplexity=np.sqrt(f_10fps_sc[n].shape[1]),
                                  theta=0.5, rand_seed=config.random_state)
            trained_tsne.append(trained_tsne_i)
            logger.info('Done embedding into 3 D.')
    if comp == 1:
        scaler = StandardScaler()
        scaler.fit(f_10fps.T)  # TODO: HIGH: variable `f_10fps` referenced before assignment. Error in logic above? ########################## IMPORTANT ###############################
        f_10fps_sc = scaler.transform(f_10fps.T).T
        logger.info(f'Training t-SNE to embed {f_10fps_sc.shape[1]} instances'
                                 f'from {f_10fps_sc.shape[0]} D into 3 D from a total of {len(data)} CSV files...')
        trained_tsne = tsne(f_10fps_sc.T, dimensions=3, perplexity=np.sqrt(f_10fps_sc.shape[1]),
                            theta=0.5, rand_seed=config.random_state)  # TODO: low: move "rand_seed" to a config file instead of hiding here as magic variable
        logger.info(f'{inspect.stack()[0][3]}::Done embedding into 3 D.')
    return f_10fps, f_10fps_sc, trained_tsne, scaler
def bsoid_tsne_voc(data: list, bodyparts=config.BODYPARTS_VOC_LEGACY, fps=config.video_fps, comp: int = config.compile_CSVs_for_training, tsne_params=config.TSNE_PARAMS):
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
        err = f'`data` was expected to be list of data (specifically arrays) but no data was found. ' \
              f'data = {data}'
        logger.error(err)
        raise ValueError(err)
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logger.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data[m])
        p12 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point2'):2 * bodyparts.get('Point2') + 2]
        p14 = data[m][:, 2 * bodyparts.get('Point1'):2 * bodyparts.get('Point1') + 2] - \
              data[m][:, 2 * bodyparts.get('Point4'):2 * bodyparts.get('Point4') + 2]
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
        for k in range(data_range - 1):
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
    logger.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    if comp == 0:
        f_10fps = []
        f_10fps_sc = []
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
                feats1 = np.hstack((np.mean((feats[n][0:2, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][2:6, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(feats[0]), 1)
        logger.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
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
            logger.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from CSV file {}...'.format(f_10fps_sc[n].shape[1], f_10fps_sc[n].shape[0],
                                                               n + 1))
            trained_tsne_i = TSNE(perplexity=np.sqrt(f_10fps_sc[n].shape[1]),
                                  early_exaggeration=16,  # early exaggeration alpha 16 is good
                                  learning_rate=max(200, f_10fps_sc[n].shape[1] / 16),  # alpha*eta = n
                                  **tsne_params).fit_transform(f_10fps_sc[n].T)
            trained_tsne.append(trained_tsne_i)
            logger.info('Done embedding into 3 D.')
        if comp == 1:
            scaler = StandardScaler()
            scaler.fit(f_10fps.T)
            f_10fps_sc = scaler.transform(f_10fps.T).T
            logger.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from a total of {} CSV files...'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
                                                                           len(data)))
            trained_tsne = TSNE(perplexity=np.sqrt(f_10fps_sc.shape[1]),  # perplexity scales with sqrt, power law
                                early_exaggeration=16,  # early exaggeration alpha 16 is good
                                learning_rate=max(200, f_10fps_sc.shape[1] / 16),  # alpha*eta = n
                                **tsne_params).fit_transform(f_10fps_sc.T)
            logger.info('Done embedding into 3 D.')
    return f_10fps, f_10fps_sc, trained_tsne


def train_emgmm_with_learned_tsne_space(trained_tsne_array, comp=config.compile_CSVs_for_training, emgmm_params=config.EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne_array: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logger.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne_array.shape))
        gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne_array)
        logger.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne_array.shape))
        assigns = gmm.predict(trained_tsne_array)
    else:
        assigns = []
        for i in tqdm(range(len(trained_tsne_array))):
            logger.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne_array[i].shape))
            gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne_array[i])
            logger.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne_array[i].shape))
            assign = gmm.predict(trained_tsne_array[i])
            assigns.append(assign)
    logger.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne_array.shape))
    uk = list(np.unique(assigns))
    assignments_list = []
    for i in assigns:
        idx_value = uk.index(i)
        assignments_list += [idx_value]
    assignments = np.array(assignments_list)
    return assignments

def bsoid_svm_py(feats, labels, comp: int = config.compile_CSVs_for_training, holdout_pct: float = config.holdout_percent, cv_it: int = config.crossvalidation_k, svm_params: dict = config.SVM_PARAMS):
    # TODO: low: depending on COMP value, could return two lists or a classifier and a list...consistency?!!!!
    """
    Train SVM classifier
    
    :param comp:
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param holdout_pct: (float) Test partition ratio for validating SVM performance in GLOBAL_CONFIG
    :param cv_it: (int) iterations for cross-validation in GLOBAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            feats.T, labels.T, test_size=holdout_pct, random_state=config.random_state)  # TODO: med: MAGIC VARIABLES -- `random_state`
        logger.info(f'Training SVM on randomly partitioned {(1-holdout_pct)*100}% of training data...')
        classifier = SVC(**svm_params)
        classifier.fit(feats_train, labels_train)
        logger.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
        logger.info(f'Predicting randomly sampled (non-overlapped) assignments '
                     f'using the remaining {holdout_pct * 100}%...')
        scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=config.crossvalidation_n_jobs)
        time_str = time.strftime("_%Y%m%d_%H%M")
        if config.PLOT_TRAINING:
            np.set_printoptions(precision=2)
            titles_options = [("Non-normalized confusion matrix", None),
                              ("Normalized confusion matrix", 'true')]
            titlenames = ["counts", "norm"]
            j = 0
            for title, normalize in titles_options:
                display = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                cmap=plt.cm.Blues, normalize=normalize)
                display.ax_.set_title(title)
                print(title)
                print(display.confusion_matrix)
                my_file = f'confusion_matrix_{titlenames[j]}'
                display.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}{time_str}.svg'))
                j += 1
            plt.show()
    else:
        classifier, scores = [], []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(
                feats[i].T, labels[i].T, test_size=holdout_pct, random_state=config.random_state)
            logger.info(f'Training SVM on randomly partitioned {(1 - holdout_pct) * 100}% of training data...')
            clf = SVC(**svm_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logger.info(f'Done training SVM mapping {feats_train.shape} features to {labels_train.shape} assignments.')
            logger.info(f'Predicting randomly sampled (non-overlapped) assignments using '
                         f'the remaining {holdout_pct * 100}%...')
            sc = cross_val_score(classifier, feats_test, labels_test, cv=cv_it,
                                 n_jobs=-1)  # TODO: low: `sc` unused variable
            time_str = time.strftime("_%Y%m%d_%H%M")
            if config.PLOT_TRAINING:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                titlenames = ["counts", "norm"]
                for title, normalize in titles_options:
                    display = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                    cmap=plt.cm.Blues, normalize=normalize)
                    display.ax_.set_title(title)
                    print(title)
                    print(display.confusion_matrix)
                    my_file = f'confusion_matrix_clf{i}_{titlenames[j]}'
                    display.figure_.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file}{time_str}.svg'))
                    j += 1
                plt.show()
    logger.info('Scored cross-validated SVM performance.'.format(feats_train.shape,
                                                                  labels_train.shape))  # TODO: low: .format() called but variables never used
    return classifier, scores


### Legacy functions -- keep them for now

def bsoid_nn_voc(feats, labels, comp: int = config.compile_CSVs_for_training, hldout: float = config.holdout_percent, cv_it=config.crossvalidation_k, mlp_params=config.MLP_PARAMS):
    # WARNING: DEPRECATION IMMINENT
    replacement_func = train_mlp_classifier_voc
    warnings.warn(f'This function will be deprecated in the future. If you still need this function to use, '
                  f'think about using {replacement_func.__qualname__} instead.')
    return replacement_func(feats, labels, comp, hldout, cv_it, mlp_params)
def bsoid_gmm_pyvoc(trained_tsne_array, comp=config.compile_CSVs_for_training, emgmm_params=config.EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne_array: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    replacement_func = train_emgmm_with_learned_tsne_space
    warnings.warn(f'This function will be deprecated in the future. To resolve this warning, replace this '
                  f'function with {replacement_func.__qualname__} instead.')
    return replacement_func(trained_tsne_array, comp=config.compile_CSVs_for_training, emgmm_params=config.EMGMM_PARAMS)
def bsoid_feats_umapapp(data: list, fps: int = config.video_fps) -> Tuple:
    # WARNING: DEPRECATION IMMINENT
    replacement_func = train_umap_unsupervised_with_xy_features_umapapp
    warnings.warn(f'DEPRECATION WARNING. This function will be deprecated in favour of a more clear '
                  f'and concise function. Current replacement is: {replacement_func.__qualname__}. '
                  f'This function only still exists to ensure dependencies aren\'t broken on updating entire module')
    return replacement_func(data, fps)


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
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}')
    # Get data
    filenames, training_data, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_BASEPATH_and_process_data(train_folders)
    if len(filenames) == 0:
        raise ValueError('Zero training folders were specified. Check your config file!!!!')
    if len(filenames[0]) == 0:
        logger.error('train.py::main()::Zero filenames were found.')
        raise ValueError(f'UNEXPECTEDLY ZERO FILES. ARE YOU SURE BASE_PATH IS SET CORRECTLY? OR GLOB PATH CHECKING MAY NEED SOME WORK')  # TODO: clarify

    # Train TSNE
    features_10fps, features_10fps_scaled, trained_tsne_list, scaler = bsoid_tsne_py(training_data)

    # Train GMM
    # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below
    gmm_assignments = train_emgmm_with_learned_tsne_space(trained_tsne_list)

    # Train SVM
    classifier, scores = bsoid_svm_py(features_10fps_scaled, gmm_assignments)

    # Plot to view progress if necessary
    if config.PLOT_TRAINING:
        visuals.plot_classes_EMGMM_assignments(trained_tsne_list, gmm_assignments, True)  # TODO: HIGH: save fig to file is a magic variable
        visuals.plot_accuracy_SVM(scores)
        visuals.plot_accuracy_SVM(features_10fps, gmm_assignments)
    return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores


def main_umap(train_folders: list):
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}')

    time_str = time.strftime("_%Y%m%d_%H%M")
    filenames, training_data, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_BASEPATH_and_process_data(train_folders)
    features_10fps, features_10fps_scaled = train_umap_unsupervised_with_xy_features_umapapp(training_data)

    # Train UMAP (unsupervised) given a set of features based on (x,y) positions
    trained_umap, umap_embeddings = bsoid_umap_embed_umapapp(features_10fps_scaled)
    # Train HDBSCAN (unsupervised) given learned UMAP space
    hdb_assignments, soft_clusters, soft_assignments = bsoid_hdbscan_umapapp(umap_embeddings)

    nn_classifier, scores, nn_assignments = bsoid_nn_appumap(features_10fps, soft_assignments)

    if config.PLOT_GRAPHS:
        fig1 = visuals.plot_classes_bsoidumap(umap_embeddings[hdb_assignments >= 0], hdb_assignments[hdb_assignments >= 0])
        my_file1 = 'hdb_soft_assignments'
        fig1.savefig(os.path.join(config.OUTPUT_PATH, f'{my_file1}{time_str}.svg'))  # fig1.savefig(os.path.join(config.OUTPUT_PATH, str.join('', (my_file1, time_str, '.svg'))))
        visuals.plot_accuracy_umap(scores)

    return features_10fps, features_10fps_scaled, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, \
        nn_classifier, scores, nn_assignments


def main_voc(train_folders: list):
    if not isinstance(train_folders, list):
        raise ValueError(f'`train_folders` arg was expected to be list but instead found '
                         f'type: {type(train_folders)} (value:  {train_folders}')

    filenames, training_data, perc_rect = likelihoodprocessing.\
        import_csvs_data_from_folders_in_BASEPATH_and_process_data(train_folders)

    # Train T-SNE
    f_10fps, f_10fps_sc, trained_tsne = bsoid_tsne_voc(training_data)

    # Train GMM
    # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below
    gmm_assignments = train_emgmm_with_learned_tsne_space(trained_tsne)

    # Train classifier
    classifier, scores = train_mlp_classifier_voc(f_10fps, gmm_assignments)
    if config.PLOT_TRAINING:
        visuals.plot_classes_bsoidvoc(trained_tsne, gmm_assignments)
        visuals.plot_accuracy_bsoidvoc(scores)
        visuals.plot_feats_bsoidvoc(f_10fps, gmm_assignments)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores


if __name__ == '__main__':
    main_py(config.TRAIN_FOLDERS)
    pass
