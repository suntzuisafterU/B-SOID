""" ################# BSOID_PY
A master that runs BOTH
1. Training a unsupervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""
""" ######### BSOID_UMAP
A master that runs BOTH
1. Training a unsupervised + supervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""
"""######## BSOID_VOC
A master that runs BOTH:
    1. Training a unsupervised machine learning model based on patterns in spatio-temporal (x,y) changes.
    2. Predicting new behaviors using (x,y) based on learned classifier.
"""

from typing import Any, List, Tuple
import itertools
import joblib
import numpy as np
import os
import pandas as pd
import time

from bsoid import classify, config, train, util
from bsoid.config import video_fps as video_fps, OUTPUT_PATH as OUTPUT_PATH, TRAIN_FOLDERS as TRAIN_FOLDERS, PREDICT_FOLDERS as PREDICT_FOLDERS

logger = config.bsoid_logger

""" py
    Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
1. original features (number of training data points by 7 dimensions, columns 1-7)
2. embedded features (number of training data points by 3 dimensions, columns 8-10)
3. em-gmm assignments (number of training data points by 1, columns 11)
Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
:param train_folders: list, folders to build behavioral model on
:returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
"""
def build_py(train_folders) -> Tuple[Any, Any, Any, Any, Any, Any]:
    f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores = train.main_py(train_folders)
    alldata = np.concatenate([f_10fps.T, trained_tsne, gmm_assignments.reshape(len(gmm_assignments), 1)], axis=1)
    multi_index_columns = pd.MultiIndex.from_tuples([
        ('Features', 'Relative snout to forepaws placement'),
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
    training_data = pd.DataFrame(alldata, columns=multi_index_columns)
    time_str = time.strftime("_%Y%m%d_%H%M")
    # Write training data to csv
    training_data.to_csv(os.path.join(OUTPUT_PATH, f'bsoid_trainlabels_10Hz{time_str}.csv'),
                         index=True, chunksize=10000, encoding='utf-8')
    
    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.csv'), 'wb') as f:
        joblib.dump([classifier, scaler], f)
    
    logger.info('Saved stuff...elaborate on this message later. Check build() or something like it.')  # TODO: elaborate on log message
    return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores
def build_umap(train_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
    Automatically saves single CSV file containing training outputs.
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    """
    # import bsoid_umap.train
    # from bsoid_umap.utils.statistics import feat_dist
    f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments = train.main_umap(train_folders)
    timestr = time.strftime("_%Y%m%d_%H%M")
    feat_range, feat_med, p_cts, edges = util.statistics.feat_dist(f_10fps)  #     feat_range, feat_med, p_cts, edges = feat_dist(f_10fps)
    f_range_df = pd.DataFrame(feat_range, columns=['5%tile', '95%tile'])
    f_med_df = pd.DataFrame(feat_med, columns=['median'])
    f_pcts_df = pd.DataFrame(p_cts)
    f_pcts_df.columns = pd.MultiIndex.from_product([f_pcts_df.columns, ['prob']])
    f_edge_df = pd.DataFrame(edges)
    f_edge_df.columns = pd.MultiIndex.from_product([f_edge_df.columns, ['edge']])
    f_dist_data = pd.concat((f_range_df, f_med_df, f_pcts_df, f_edge_df), axis=1)
    # Write data to csv
    f_dist_data.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_featdist_10Hz{timestr}.csv')),
                       index=True, chunksize=10000, encoding='utf-8')

    length_nm, angle_nm, disp_nm = [], [], []
    for i, j in itertools.combinations(range(0, int(np.sqrt(f_10fps.shape[0]))), 2):
        length_nm.append(['distance between points:', i+1, j+1])
        angle_nm.append(['angular change for points:', i+1, j+1])
    for i in range(int(np.sqrt(f_10fps.shape[0]))):
        disp_nm.append(['displacement for point:', i+1, i+1])
    mcol = np.vstack((length_nm, angle_nm, disp_nm))
    feat_nm_df = pd.DataFrame(f_10fps.T, columns=mcol)
    umaphdb_data = np.concatenate([umap_embeddings, hdb_assignments.reshape(len(hdb_assignments), 1),
                              soft_assignments.reshape(len(soft_assignments), 1),
                              nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
    multi_index_columns = pd.MultiIndex.from_tuples([('UMAP embeddings', 'Dimension 1'), ('', 'Dimension 2'),
                                           ('', 'Dimension 3'), ('HDBSCAN', 'Assignment No.'),
                                           ('HDBSCAN*SOFT', 'Assignment No.'), ('Neural Net', 'Assignment No.')],
                                          names=['Type', 'Frame@10Hz'])
    umaphdb_df = pd.DataFrame(umaphdb_data, columns=multi_index_columns)
    training_data = pd.concat((feat_nm_df, umaphdb_df), axis=1)
    soft_clust_prob = pd.DataFrame(soft_clusters)
    training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_trainlabels_10Hz', timestr, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
    soft_clust_prob.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labelprob_10Hz', timestr, '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', config.MODEL_NAME, '.sav'))), 'wb') as f:
        joblib.dump([f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters,
                     nn_classifier, scores, nn_assignments], f)
    logger.info('Saved. Expand on this info message.')
    return f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
           scores, nn_assignments
def build_voc(train_folders) -> Tuple:
    """
    Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. embedded features (number of training data points by 3 dimensions, columns 8-10)
    3. em-gmm assignments (number of training data points by 1, columns 11)
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_voc.train
    """
    # import bsoid_voc.train
    f_10fps, trained_tsne, gmm_assignments, classifier, scores = train.main_voc(train_folders)
    alldata = np.concatenate([f_10fps.T, trained_tsne, gmm_assignments.reshape(len(gmm_assignments), 1)], axis=1)
    micolumns = pd.MultiIndex.from_tuples([('Features', 'Distance between points 1 & 5'),
                                           ('', 'Distance between points 1 & 8'),
                                           ('', 'Angle change between points 1 & 2'),
                                           ('', 'Angle change between points 1 & 4'),
                                           ('', 'Point 3 displacement'),
                                           ('', 'Point 7 displacement'),
                                           ('Embedded t-SNE', 'Dimension 1'),
                                           ('', 'Dimension 2'),
                                           ('', 'Dimension 3'),
                                           ('EM-GMM', 'Assignment No.')],
                                          names=['Type', 'Frame@10Hz'])
    training_data = pd.DataFrame(alldata, columns=micolumns)
    timestr = time.strftime("_%Y%m%d_%H%M")
    training_data.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_trainlabels_10Hz{timestr}.csv')),
                         index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.csv'), 'wb') as f:
        joblib.dump(classifier, f)
    logger.info('Saved.')  # TODO: add specificity to log
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores


def run_py(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in video_fps, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. SVM predicted labels (number of training data points by 1, columns 8)
    """
    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'rb') as fr:
        behv_model, scaler = joblib.load(fr)
    data_new, feats_new, labels_fslow, labels_fshigh = classify.main_py(predict_folders, scaler, video_fps, behv_model)
    filenames = []
    all_df = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        file_names_csvs: List[str] = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_basepath(fd)
        for j, filename in enumerate(file_names_csvs):
            logger.info(f'Importing CSV file {j+1} from folder {i+1}')
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames.append(filename)
            all_df.append(curr_df)
    for i in range(len(feats_new)):
        alldata = np.concatenate([feats_new[i].T, labels_fslow[i].reshape(len(labels_fslow[i]), 1)], axis=1)
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
        runlen_df1, dur_stats1, df_tm1 = util.statistics.main_py(labels_fslow[i])
        # if PLOT_TRAINING:
        #     plot_tmat(df_tm1, video_fps)
        runlen_df1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_10Hz', timestr, csvname, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        dur_stats1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_10Hz', timestr, csvname, '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_10Hz', timestr, csvname, '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
        labels_fshigh_pad = np.pad(labels_fshigh[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_fshigh_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''  # TODO: low: address duplicate line here and below
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_', str(video_fps), 'Hz', timestr, csvname,
                                                                '.csv')))),
                       index=True, chunksize=10000, encoding='utf-8')
        runlen_df2, dur_stats2, df_tm2 = util.statistics.main(labels_fshigh[i])
        runlen_df2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(video_fps), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        dur_stats2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_', str(video_fps), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_', str(video_fps), 'Hz', timestr, csvname,
                                                               '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, 'bsoid_predictions.sav'), 'wb') as file_names_csvs:
        joblib.dump([labels_fslow, labels_fshigh], file_names_csvs)
    logger.info('All saved. Expand on this info message later.')
    return data_new, feats_new, labels_fslow, labels_fshigh
def run_umap(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns data_new, fs_labels: see bsoid_umap.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs.
    """
    time_str = time.strftime("_%Y%m%d_%H%M")
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', config.MODEL_NAME, '.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, \
        nn_classifier, scores, nn_assignments = joblib.load(fr)
    data_new, fs_labels = classify.main_umap(predict_folders, video_fps, nn_classifier)
    filenames_list = []
    all_df = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        f = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_basepath(fd)
        for j, filename in enumerate(f):
            logger.info(f'Importing CSV file {j+1} from folder {i+1}')
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames_list.append(filename)
            all_df.append(curr_df)

    for i in range(len(fs_labels)):
        csvname = os.path.basename(filenames_list[i]).rpartition('.')[0]
        fs_labels_pad = np.pad(fs_labels[i], (6, 0), 'edge')
        df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_labels_{video_fps}Hz{time_str}{csvname}.csv')),
                       index=True, chunksize=10000, encoding='utf-8')
        runlen_df, dur_stats, df_tm = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(fs_labels[i])
        runlen_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(video_fps), 'Hz', time_str, csvname,
                                                                  '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        dur_stats.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_', str(video_fps), 'Hz', time_str, csvname,
                                                                  '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        df_tm.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_', str(video_fps), 'Hz', time_str, csvname,
                                                              '.csv')))),
                     index=True, chunksize=10000, encoding='utf-8')
        if config.PLOT_TRAINING:
            fig = util.visuals.plot_tmat(df_tm)
            my_file = 'transition_matrix'
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, str(video_fps), 'Hz', time_str, csvname, '.svg'))))
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_predictions', time_str, '.sav'))), 'wb') as f:
        joblib.dump([data_new, fs_labels], f)
    logger.info('All saved.')
    return data_new, fs_labels
def run_voc(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_voc.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in video_fps, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. Neural net predicted labels (number of training data points by 1, columns 8)
    """
    # import bsoid_voc.classify; from bsoid_voc.utils.likelihoodprocessing import get_filenames; import bsoid_voc.utils.statistics; from bsoid_voc.utils.visuals import plot_tmat

    with open(os.path.join(OUTPUT_PATH, 'bsoid_'+config.MODEL_NAME+'.sav'), 'rb') as fr:
        behv_model = joblib.load(fr)

    data_new, feats_new, labels_fslow, labels_fshigh = classify.main_voc(predict_folders, video_fps, behv_model)
    filenames = []
    all_df = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        f = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_basepath(fd)
        for j, filename in enumerate(f):
            logger.info('Importing CSV file {} from folder {}'.format(j + 1, i + 1))
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames.append(filename)
            all_df.append(curr_df)
    for i in range(0, len(feats_new)):
        alldata = np.concatenate([feats_new[i].T, labels_fslow[i].reshape(len(labels_fslow[i]), 1)], axis=1)
        micolumns = pd.MultiIndex.from_tuples([('Features', 'Distance between points 1 & 5'),
                                               ('', 'Distance between points 1 & 8'),
                                               ('', 'Angle change between points 1 & 2'),
                                               ('', 'Angle change between points 1 & 4'),
                                               ('', 'Point 3 displacement'), ('', 'Point 7 displacement'),
                                               ('Neural net classifier', 'B-SOiD labels')],
                                              names=['Type', 'Frame@10Hz'])
        predictions = pd.DataFrame(alldata, columns=micolumns)
        timestr = time.strftime("_%Y%m%d_%H%M")
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        predictions.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_10Hz', timestr, csvname, '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
        # runlen_df1, dur_stats1, df_tm1 = bsoid_voc.utils.statistics.main(labels_fslow[i])
        runlen_df1, dur_stats1, df_tm1 = util.statistics.main(labels_fslow[i])
        if config.PLOT_TRAINING:
            util.visuals.plot_tmat(df_tm1, video_fps)
        runlen_df1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_10Hz', timestr, csvname, '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        dur_stats1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_10Hz', timestr, csvname, '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm1.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_10Hz', timestr, csvname, '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
        labels_fshigh_pad = np.pad(labels_fshigh[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_fshigh_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_', str(video_fps), 'Hz', timestr, csvname,
                                                                '.csv')))),
                       index=True, chunksize=10000, encoding='utf-8')
        runlen_df2, dur_stats2, df_tm2 = util.statistics.main(labels_fshigh[i])
        runlen_df2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(video_fps), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        dur_stats2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_', str(video_fps), 'Hz', timestr, csvname,
                                                                   '.csv')))),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_', str(video_fps), 'Hz', timestr, csvname,
                                                               '.csv')))),
                      index=True, chunksize=10000, encoding='utf-8')
    logger.info('All saved.')
    return data_new, feats_new, labels_fslow, labels_fshigh


def main_py(train_folders, predict_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    :returns feats_new, labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores = build_py(train_folders)
    data_new, feats_new, labels_fslow, labels_fshigh = run_py(predict_folders)
    return f_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh
def main_umap(train_folders, predict_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
    :returns feats_new, fs_labels: see bsoid_umap.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
    scores, nn_assignments = build_umap(train_folders)
    data_new, fs_labels = run_umap(predict_folders)
    return f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
           scores, nn_assignments, data_new, fs_labels
def main_voc(train_folders, predict_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_voc.train
    :returns feats_new, labels_fslow, labels_fshigh: see bsoid_voc.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, trained_tsne, gmm_assignments, classifier, scores = build_voc(train_folders)
    data_new, feats_new, labels_fslow, labels_fshigh = run_voc(predict_folders)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh


# def main(train_folders: List[str], predict_folders: List[str]):
#
#     # Build
#
#     # run_
#
#     # Return
#     return


## TODO: Q: retrain() was commented-out by authors...not sure if keep or delete
# def retrain_umap(train_folders):
#     """
#     :param train_folders: list, folders to build behavioral model on
#     :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
#     Automatically saves single CSV file containing training outputs.
#     Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
#     """
#     with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', config.MODEL_NAME, '.sav'))), 'rb') as fr:
#         f_10fps, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, \
#         nn_assignments = joblib.load(fr)
#     from bsoid_umap.util.videoprocessing import vid2frame
#     vid2frame(VID_NAME, f_10fps[ID], video_fps, FRAME_DIR)
#     labels_df = pd.read_csv('/Users/ahsu/Sign2Speech/Notebook/labels.csv', low_memory=False)
#
#     import bsoid_umap.retrain
#     f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments = bsoid_umap.train.main(train_folders)
#     alldata = np.concatenate([umap_embeddings, nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
#     micolumns = pd.MultiIndex.from_tuples([('UMAP embeddings', 'Dimension 1'), ('', 'Dimension 2'),
#                                            ('', 'Dimension 3'), ('Neural Net', 'Assignment No.')],
#                                           names=['Type', 'Frame@10Hz'])
#     training_data = pd.DataFrame(alldata, columns=micolumns)
#     timestr = time.strftime("_%Y%m%d_%H%M")
#     training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_trainlabels_10Hz', timestr, '.csv')))),
#                          index=True, chunksize=10000, encoding='utf-8')
#     with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', config.MODEL_NAME, '.sav'))), 'wb') as f:
#         joblib.dump([f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments], f)
#     logger.info('Saved.')
#     return f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments


if __name__ == "__main__":  # py
    main_py(TRAIN_FOLDERS, PREDICT_FOLDERS)


# if __name__ == "__main__":  # umap
#     f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
#         scores, nn_assignments, data_new, fs_labels = main_umap(TRAIN_FOLDERS, PREDICT_FOLDERS)
#
#
# if __name__ == "__main__":  # voc
#     f_10fps, trained_tsne, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh \
#         = main_voc(TRAIN_FOLDERS, PREDICT_FOLDERS)
