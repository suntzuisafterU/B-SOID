"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD behavioral model.

Potential abbreviations:
    sn: snout
    pt: proximal tail
"""
import os
import sys

project_relative_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_relative_path)
# print(sys.path)


# # # General imports # # #
from bhtsne import tsne as TSNE_bht
from sklearn.metrics import plot_confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Any, List, Tuple
from tqdm import tqdm
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# # # B-SOID imports # # #BN
# import bsoid
from bsoid_py.utils import likelihoodprocessing
from bsoid.util.likelihoodprocessing import boxcar_center
from bsoid_py.config.LOCAL_CONFIG import BODYPARTS, COMP, FPS, OUTPUT_PATH, PLOT_TRAINING, TRAIN_FOLDERS
from bsoid_py.config.GLOBAL_CONFIG import CV_IT, EMGMM_PARAMS, HLDOUT, SVM_PARAMS
from bsoid_py.utils.visuals import plot_accuracy, plot_classes, plot_feats


def hard_coded_feature_extraction(data, bodyparts: dict, win_len: int) -> List:
    """

    :param data: list of 3D array
    """
    features = []
    for m in range(len(data)):
        logging.info(f'Extracting features from CSV file {m+1}...')
        data_range = len(data[m])
        fpd = data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1'):2 * bodyparts.get('Forepaw/Shoulder1') + 2] - data[m][:, 2 * bodyparts.get('Forepaw/Shoulder2'):2 * bodyparts.get('Forepaw/Shoulder2') + 2]
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
            sn_pt_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(sn_pt[k, :], sn_pt[k + 1, :])))
            sn_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1] -
                data[m][k, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1])
            pt_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1] -
                data[m][k, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1])
        sn_pt_ang_smth = boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = boxcar_center(sn_disp, win_len)
        pt_disp_smth = boxcar_center(pt_disp, win_len)
        features.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:], sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logging.info(f'Done extracting features from a total of {len(data)} training CSV files.')
    return features


def bsoid_tsne(data: list, bodyparts=BODYPARTS, fps=FPS) -> Tuple[List, List, TSNE_bht, object]:
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
    # win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    win_len: int = int(round(0.05 / (1 / fps)) * 2 - 1)

    # Extract features
    features = hard_coded_feature_extraction(data, bodyparts, win_len)

    features_10fps, features_10fps_scaled, trained_tsne = [], [], []
    # Loop over the number of features (as found from the hard-coded extraction)
    for i in range(len(features)):
        features_i = np.zeros(len(data[i]))

        for j in range(round(fps / 10) - 1, len(features[i][0]), round(fps / 10)):
            if j > round(fps / 10) - 1:
                features_i = np.concatenate((features_i.reshape(features_i.shape[0], features_i.shape[1]),
                                         np.hstack((np.mean((features[i][0:4, range(j - round(fps / 10), j)]), axis=1),
                                                    np.sum((features[i][4:7, range(j - round(fps / 10), j)]),
                                                           axis=1))).reshape(len(features[0]), 1)), axis=1)
            else:
                features_i = np.hstack((np.mean((features[i][0:4, range(j - round(fps / 10), j)]), axis=1),
                                        np.sum((features[i][4:7, range(j - round(fps / 10), j)]), axis=1))).reshape(len(features[0]), 1)
        logging.info(f'Done integrating features into 100ms bins from CSV file {i+1}.')

        # if comp == 1:
        if i > 0:
            features_10fps = np.concatenate((features_10fps, features_i), axis=1)
        else:
            features_10fps = features_i
        # else:
        #     features_10fps.append(feats1)
        #     scaler = StandardScaler()
        #     scaler.fit(feats1.T)
        #     feats1_stnd = scaler.transform(feats1.T).T
        #     features_10fps_sc.append(feats1_stnd)
        #     logging.info(f'Training t-SNE to embed {features_10fps_sc[i].shape[1]} instances from '
        #                  f'{features_10fps_sc[i].shape[0]} D into 3 D from CSV file {i + 1}...')
        #     trained_tsne_i = TSNE_bht(features_10fps_sc[i].T, dimensions=3, perplexity=np.sqrt(features_10fps_sc[i].shape[1]),
        #                               theta=0.5, rand_seed=23)
        #     trained_tsne_list.append(trained_tsne_i)
        #     logging.info('Done embedding into 3 D.')

    # if comp == 1:
    scaler = StandardScaler()
    scaler.fit(features_10fps.T)
    features_10fps_scaled = scaler.transform(features_10fps.T).T
    logging.info(f'Training t-SNE to embed {features_10fps_scaled.shape[1]} instances '
                 f'from {features_10fps_scaled.shape[0]} D into 3 D from a total of {len(data)} CSV files...')
    trained_tsne = TSNE_bht(features_10fps_scaled.T,
                            dimensions=3, perplexity=np.sqrt(features_10fps_scaled.shape[1]),
                            theta=0.5, rand_seed=23)  # TODO: low: move "rand_seed" to a config file instead of hiding here as magic variable
    logging.info('Done embedding into 3 D.')
    return features_10fps, features_10fps_scaled, trained_tsne, scaler


def bsoid_gmm(trained_tsne, comp=COMP, emgmm_params=EMGMM_PARAMS) -> np.ndarray:
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne.shape))
        gmm = GaussianMixture(**emgmm_params).fit(trained_tsne)
        logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
        assigns = gmm.predict(trained_tsne)
    else:
        assigns = []
        for i in tqdm(range(len(trained_tsne))):
            logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne[i].shape))
            gmm = GaussianMixture(**emgmm_params).fit(trained_tsne[i])
            logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne[i].shape))
            assign = gmm.predict(trained_tsne[i])
            assigns.append(assign)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
    uk = list(np.unique(assigns))

    assignments_list = []
    for i in assigns:
        indexVal = uk.index(i)
        assignments_list.append(indexVal)

    # Coerce to array and return
    assignments = np.array(assignments_list)
    return assignments


def bsoid_svm(features, labels, comp=COMP, hldout=HLDOUT, cv_it=CV_IT, svm_params=SVM_PARAMS):
    """
    Trains SVM classifier
    :param features: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param hldout: scalar, test partition ratio for validating SVM performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(
            features.T, labels.T, test_size=hldout, random_state=23)
        logging.info(f'Training SVM on randomly partitioned {(1 - hldout) * 100}% of training data...')
        classifier = SVC(**svm_params)
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
            titlenames = ["counts", "norm"]
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
        for i in range(len(features)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(
                features[i].T, labels[i].T, test_size=hldout, random_state=23)
            logging.info(f'Training SVM on randomly partitioned {(1 - hldout) * 100}% of training data...')
            clf = SVC(**svm_params)
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
                titlenames = ["counts", "norm"]
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


def run_train_py(train_folders: list):
    """
    :param train_folders: list, training data folders
    :return f_10fps: 2D array, features
    :return trained_tsne: 2D array, trained t-SNE space
    :return gmm_assignments: Converged EM-GMM group assignments
    :return classifier: obj, SVM classifier
    :return scores: 1D array, cross-validated accuracy
    """
    assert isinstance(train_folders, list)
    # Get data
    filenames, training_data, perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_BASEPATH(train_folders)
    if len(filenames) == 0:
        raise ValueError('Zero training folders were specified. Check your config file!!!!')
    if len(filenames[0]) == 0:
        logging.error('train.py::main()::Zero filenames were found.')
        raise ValueError('UNEXPECTEDLY ZERO FILES. ARE YOU SURE BASE_PATH IS SET CORRECTLY? OR GLOB PATH CHECKING MAY NEED SOME WORK')

    # Train TSNE
    features_10fps, features_10fps_scaled, trained_tsne_list, scaler = bsoid_tsne(training_data)

    # Train GMM
    gmm_assignments = bsoid_gmm(trained_tsne_list)

    # Train SVM
    classifier, scores = bsoid_svm(features_10fps_scaled, gmm_assignments)

    # Plot to view progress if necessary
    if PLOT_TRAINING:
        plot_classes(trained_tsne_list, gmm_assignments)
        plot_accuracy(scores)
        plot_feats(features_10fps, gmm_assignments)
    return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores


if __name__ == '__main__':
    # import sys
    # print('SYS PATH:', sys.path)
    run_train_py(TRAIN_FOLDERS)

