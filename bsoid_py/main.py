"""
A master that runs BOTH
1. Training a unsupervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""

import joblib
import logging
import numpy as np
import os
import pandas as pd
import time

from bsoid.config import VIDEO_FPS as FPS, MODEL_NAME, OUTPUT_PATH, TRAIN_FOLDERS, PREDICT_FOLDERS
# from bsoid.config.GLOBAL_CONFIG import CV_IT, EMGMM_PARAMS, HLDOUT, SVM_PARAMS
import bsoid


def build(train_folders):
    """
        Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. embedded features (number of training data points by 3 dimensions, columns 8-10)
    3. em-gmm assignments (number of training data points by 1, columns 11)
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    """

    f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores = bsoid.train.get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(train_folders)
    alldata = np.concatenate([f_10fps.T, trained_tsne, gmm_assignments.reshape(len(gmm_assignments), 1)], axis=1)
    micolumns = pd.MultiIndex.from_tuples([('Features', 'Relative snout to forepaws placement'),
                                           ('', 'Relative snout to hind paws placement'),
                                           ('', 'Inter-forepaw distance'),
                                           ('', 'Body length'),
                                           ('', 'Body angle'),
                                           ('', 'Snout displacement'),
                                           ('', 'Tail-base displacement'),
                                           ('Embedded t-SNE', 'Dimension 1'),
                                           ('', 'Dimension 2'),
                                           ('', 'Dimension 3'),
                                           ('EM-GMM', 'Assignment No.')],
                                          names=['Type', 'Frame@10Hz'])
    training_data = pd.DataFrame(alldata, columns=micolumns)
    timestr = time.strftime("_%Y%m%d_%H%M")
    training_data.to_csv(os.path.join(OUTPUT_PATH, 'bsoid_trainlabels_10Hz'+timestr+'.csv'),
                         index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, '.sav'))), 'wb') as f:
        joblib.dump([classifier, scaler], f)
    logging.info('Saved.')  # TODO: elaborate on log message
    return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores


def run(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in FPS, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. SVM predicted labels (number of training data points by 1, columns 8)
    """

    with open(os.path.join(OUTPUT_PATH, f'bsoid_{MODEL_NAME}.sav'), 'rb') as fr:
        behv_model, scaler = joblib.load(fr)
    data_new, feats_new, labels_frameshifted_low, labels_frameshifted_high = bsoid.classify.main_py(predict_folders, scaler, FPS, behv_model)
    filenames = []
    all_df = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        f = bsoid.util.likelihoodprocessing.get_filenames(fd)
        for j, filename in enumerate(f):
            logging.info(f'Importing CSV file {j+1} from folder {i+1}')
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames.append(filename)
            all_df.append(curr_df)
    for i in range(0, len(feats_new)):
        alldata = np.concatenate([feats_new[i].T, labels_frameshifted_low[i].reshape(len(labels_frameshifted_low[i]), 1)], axis=1)
        micolumns = pd.MultiIndex.from_tuples([('Features', 'Relative snout to forepaws placement'),
                                               ('', 'Relative snout to hind paws placement'),
                                               ('', 'Inter-forepaw distance'),
                                               ('', 'Body length'), ('', 'Body angle'), ('', 'Snout displacement'),
                                               ('', 'Tail-base displacement'),
                                               ('SVM classifier', 'B-SOiD labels')],
                                              names=['Type', 'Frame@10Hz'])
        predictions = pd.DataFrame(alldata, columns=micolumns)
        timestr = time.strftime("_%Y%m%d_%H%M")
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        predictions.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_10Hz', timestr, csvname,  '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
        runlen_df1, dur_stats1, df_tm1 = bsoid.util.statistics.main(labels_frameshifted_low[i])
        # if PLOT_TRAINING:
        #     plot_tmat(df_tm1, FPS)
        runlen_df1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_10Hz', timestr, csvname, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        dur_stats1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_10Hz', timestr, csvname, '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_10Hz', timestr, csvname, '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
        labels_frameshift_high_pad = np.pad(labels_frameshifted_high[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_frameshift_high_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''  # TODO: low: address duplicate line here and below
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_', str(FPS), 'Hz', timestr, csvname,
                                                                '.csv')))),
                       index=True, chunksize=10000, encoding='utf-8')
        runlen_df2, dur_stats2, df_tm2 = bsoid.util.statistics.main(labels_frameshifted_high[i])
        runlen_df2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(FPS), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        dur_stats2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_', str(FPS), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_', str(FPS), 'Hz', timestr, csvname,
                                                               '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, 'bsoid_predictions.sav'), 'wb') as f:
        joblib.dump([labels_frameshifted_low, labels_frameshifted_high], f)
    logging.info('All saved.')
    return data_new, feats_new, labels_frameshifted_low, labels_frameshifted_high


def main(train_folders, predict_folders):
    """
    Purpose: TODO: HIGH Importance
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    :returns feats_new, labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores = build(train_folders)
    data_new, feats_new, labels_frameshifted_low, labels_frameshifted_high = run(predict_folders)
    return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores, data_new, feats_new, labels_frameshifted_low, labels_frameshifted_high


if __name__ == "__main__":
    main(TRAIN_FOLDERS, PREDICT_FOLDERS)
