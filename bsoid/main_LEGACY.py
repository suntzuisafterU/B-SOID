"""
A master that runs BOTH
1. Training a unsupervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""

from typing import Any, List, Tuple
import inspect
import itertools
import joblib
import numpy as np
import os
import pandas as pd
import time

from bsoid import classify, classify_LEGACY, config, train, train_LEGACY, util
from bsoid.config import VIDEO_FPS, OUTPUT_PATH as OUTPUT_PATH

logger = config.initialize_logger(__name__)


###################

@config.deco__log_entry_exit(logger)
def build_py(train_folders) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
        1. original features (number of training data points by 7 dimensions, columns 1-7)
        2. embedded features (number of training data points by 3 dimensions, columns 8-10)
        3. em-gmm assignments (number of training data points by 1, columns 11)
    Automatically saves classifier in OUTPUT_PATH with MODEL_NAME in LOCAL_CONFIG
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    """

    # Do everything
    features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores = \
        train.get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(train_folders)

    # Save it all
    all_data = np.concatenate([
        features_10fps.T,
        trained_tsne,
        gmm_assignments.reshape(len(gmm_assignments), 1)
    ], axis=1)

    multi_index_columns = pd.MultiIndex.from_tuples([
        ('Features',        'Relative snout to forepaws placement'),
        ('',                'Relative snout to hind paws placement'),
        ('',                'Inter-forepaw distance'),
        ('',                'Body length'),
        ('',                'Body angle'),
        ('',                'Snout displacement'),
        ('',                'Tail-base displacement'),
        ('Embedded t-SNE',  'Dimension 1'),
        ('',                'Dimension 2'),
        ('',                'Dimension 3'),
        ('EM-GMM',          'Assignment No.')],
        names=['Type', 'Frame@10Hz'])
    df_training_data = pd.DataFrame(all_data, columns=multi_index_columns)

    # Write training data to file
    training_data_labels_10hz_csv_filename = f'BSOiD__{config.MODEL_NAME}__' \
                                             f'train_data_labels__10Hz__{config.runtime_timestr}.csv'
    df_training_data.to_csv(os.path.join(OUTPUT_PATH, training_data_labels_10hz_csv_filename),
                            index=True, chunksize=10000, encoding='utf-8')

    # Save model data to file
    model_filename = config.MODEL_FILENAME  # model_filename = f'bsoid_{config.MODEL_NAME}.sav'
    with open(os.path.join(OUTPUT_PATH, model_filename), 'wb') as model_file:
        joblib.dump([classifier, scaler_object], model_file)

    logger.error(f'{inspect.stack()[0][3]}(): Saved model to file. Form: [classifier, scaler_object]')  # TODO: see msg
    assert os.path.isfile(config.VIDEO_TO_LABEL_PATH), f'Video path got messed up before run(). path={config.VIDEO_TO_LABEL_PATH}'  # TODO: debugging effort
    return features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores


@config.deco__log_entry_exit(logger)
def run_py(predict_folders):  # TODO: HIGH: break up this function and rename. TOo many things happening.
    """
    Automatically loads classifier in OUTPUTPATH with MODELNAME in config
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in fps_video, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. SVM predicted labels (number of training data points by 1, columns 8)

    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_py.classify
    """

    # Read in existing model name
    model_file_to_read = config.MODEL_FILENAME
    path_to_model_file = os.path.join(OUTPUT_PATH, model_file_to_read)
    try:
        with open(path_to_model_file, 'rb') as fr:
            behavioural_model, scaler = joblib.load(fr)
    except FileNotFoundError as fnfe:
        file_not_found_err = f'Model not found: {path_to_model_file}.'  # TODO: HIGH: expand on err
        logger.error(file_not_found_err)
        raise FileNotFoundError(f'{file_not_found_err} // original EXCEPTION: {repr(fnfe)}.')

    #
    data_new, features_new, labels_fs_low, labels_fs_high = classify.main_py(predict_folders, scaler, VIDEO_FPS, behavioural_model)

    #
    filenames: List[str] = []
    all_dfs_list: List[pd.DataFrame] = []
    for i, folder in enumerate(predict_folders):  # Loop through folders
        file_names_csvs: List[str] = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder)
        for j, csv_filename in enumerate(file_names_csvs):
            logger.info(f'{inspect.stack()[0][3]}(): Importing CSV file {j+1} from folder {i+1}.')
            curr_df = pd.read_csv(csv_filename, low_memory=False)
            filenames.append(csv_filename)
            all_dfs_list.append(curr_df)

    for i, feature_new_i in enumerate(features_new):
        all_data: np.ndarray = np.concatenate([
            feature_new_i.T,
            labels_fs_low[i].reshape(len(labels_fs_low[i]), 1),
        ], axis=1)

        multi_index_columns = pd.MultiIndex.from_tuples([
            ('Features',        'Relative snout to forepaws placement'),
            ('',                'Relative snout to hind paws placement'),
            ('',                'Inter-forepaw distance'),
            ('',                'Body length'),
            ('',                'Body angle'),
            ('',                'Snout displacement'),
            ('',                'Tail-base displacement'),
            ('SVM classifier',  'B-SOiD labels')],
            names=['Type', 'Frame@10Hz'])
        df_predictions = pd.DataFrame(all_data, columns=multi_index_columns)

        time_str = config.runtime_timestr
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        df_predictions.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__labels__10Hz__{config.runtime_timestr}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')

        df1_runlengths, df1_duration_statistics, df1_transition_matrix = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(labels_fs_low[i])

        if config.PLOT_GRAPHS:
            fig_filename = f'{inspect.stack()[0][3]}_df1_transition_matrix'
            # TODO: resolve typing below
            util.visuals.plot_transition_matrix(
                df1_transition_matrix, config.VIDEO_FPS,
                save_fig_to_file=config.SAVE_GRAPHS_TO_FILE, fig_file_prefix=fig_filename)

        # Save (stuff?) to CSV
        df1_runlengths.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__runlengths1__10Hz__{time_str}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df1_duration_statistics.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__statistics1__10Hz__{time_str}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df1_transition_matrix.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__transitions1__10Hz__{time_str}__{csvname}.csv')),
                                     index=True, chunksize=10000, encoding='utf-8')

        #
        labels_frameshift_high_padded = np.pad(labels_fs_high[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_frameshift_high_padded, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''  # TODO: low: address duplicate line here and below
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_dfs_list[0]]
        df_xy_frameshift = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        df_xy_frameshift.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__labels__{VIDEO_FPS}Hz_{time_str}__{csvname}.csv')),
                                index=True, chunksize=10000, encoding='utf-8')

        #
        df2_runlengths, df2_duration_statistics, df2_transition_matrix = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(labels_fs_high[i])

        df2_runlengths.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__runlengths2__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df2_duration_statistics.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__statistics2__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df2_transition_matrix.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__transitions2__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')), index=True, chunksize=10000, encoding='utf-8')

    # Save _____ TODO
    with open(os.path.join(OUTPUT_PATH, 'bsoid_predictions.sav'), 'wb') as f:
        joblib.dump([labels_fs_low, labels_fs_high], f)

    logger.debug('All saved. What is "all"? Expand on this info message later.')  # TODO
    return data_new, features_new, labels_fs_low, labels_fs_high


@config.deco__log_entry_exit(logger)
def test_function_to_build_then_run_py():
    logger.debug(f'STARTING _PY TRAIN SERIES')
    build_py(config.TRAIN_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDED _PY TRAIN SERIES SUCCESSFULLY')
    logger.debug(f'STARTING _PY RUN SERIES')
    run_py(config.PREDICT_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDING _PY RUN SERIES. SUCCESS!')
    logger.debug(f'End of test.')


#####################################################################################################################################

@config.deco__log_entry_exit(logger)
def build_umap(train_folders) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
    Automatically saves single CSV file containing training outputs.
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    """
    features_10fps, features_10fps_scaled, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments = train_LEGACY.main_umap(train_folders)
    time_str = time.strftime("_%Y%m%d_%H%M")
    feat_range, feat_med, p_cts, edges = util.statistics.feat_dist(features_10fps)  # feat_range, feat_med, p_cts, edges = feat_dist(f_10fps)
    f_range_df = pd.DataFrame(feat_range, columns=['5%tile', '95%tile'])
    f_med_df = pd.DataFrame(feat_med, columns=['median'])
    f_pcts_df = pd.DataFrame(p_cts)
    f_pcts_df.columns = pd.MultiIndex.from_product([f_pcts_df.columns, ['prob']])
    f_edge_df = pd.DataFrame(edges)
    f_edge_df.columns = pd.MultiIndex.from_product([f_edge_df.columns, ['edge']])
    f_dist_data = pd.concat((f_range_df, f_med_df, f_pcts_df, f_edge_df), axis=1)
    # Write data to csv
    f_dist_data.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_featdist_10Hz_{time_str}.csv')), index=True, chunksize=10000, encoding='utf-8')
    #
    length_nm, angle_nm, disp_nm = [], [], []
    for m, n in itertools.combinations(range(0, int(np.sqrt(features_10fps.shape[0]))), 2):
        length_nm.append(['distance between points:', m+1, n+1])
        angle_nm.append(['angular change for points:', m+1, n+1])
    for i in range(int(np.sqrt(features_10fps.shape[0]))):
        disp_nm.append(['displacement for point:', i+1, i+1])
    m_col = np.vstack((length_nm, angle_nm, disp_nm))
    feat_nm_df = pd.DataFrame(features_10fps.T, columns=m_col)
    umaphdb_data = np.concatenate([umap_embeddings, hdb_assignments.reshape(len(hdb_assignments), 1), soft_assignments.reshape(len(soft_assignments), 1), nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
    multi_index_columns = pd.MultiIndex.from_tuples([
        ('UMAP embeddings', 'Dimension 1'),
        ('',                'Dimension 2'),
        ('',                'Dimension 3'),
        ('HDBSCAN',         'Assignment No.'),
        ('HDBSCAN*SOFT',    'Assignment No.'),
        ('Neural Net',      'Assignment No.')],
        names=['Type', 'Frame@10Hz'])
    df_umaphdb = pd.DataFrame(umaphdb_data, columns=multi_index_columns)

    # Add columns (add frames sideways)
    df_training_data: pd.DataFrame = pd.concat((feat_nm_df, df_umaphdb), axis=1)
    df_soft_clust_probability = pd.DataFrame(soft_clusters)

    # Save DataFrames to CSV
    df_training_data.to_csv((os.path.join(OUTPUT_PATH, f'bsoid__trainlabels__10Hz_{time_str}.csv')), index=True, chunksize=10000, encoding='utf-8')
    df_soft_clust_probability.to_csv((os.path.join(OUTPUT_PATH, f'bsoid__labelprob__10Hz_{time_str}.csv')), index=True, chunksize=10000, encoding='utf-8')

    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'wb') as f:
        joblib.dump([features_10fps, features_10fps_scaled, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments], f)

    logger.info('Saved. Expand on this info message.')
    return features_10fps, features_10fps_scaled, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments
@config.deco__log_entry_exit(logger)
def build_voc(train_folders) -> Tuple[Any, Any, Any, Any, List]:
    """
    Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. embedded features (number of training data points by 3 dimensions, columns 8-10)
    3. em-gmm assignments (number of training data points by 1 dimensions, columns 11)
    ** Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG **

    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_voc.train
    """
    # import bsoid_voc.train
    f_10fps, trained_tsne, gmm_assignments, classifier, scores = train_LEGACY.train__import_data_and_process__train_tsne__train_gmm__train_clf__voc(train_folders)
    all_data_as_array = np.concatenate([f_10fps.T, trained_tsne, gmm_assignments.reshape(len(gmm_assignments), 1)], axis=1)

    multi_index_columns = pd.MultiIndex.from_tuples([
        ('Features',        'Distance between points 1 & 5'),
        ('',                'Distance between points 1 & 8'),
        ('',                'Angle change between points 1 & 2'),
        ('',                'Angle change between points 1 & 4'),
        ('',                'Point 3 displacement'),
        ('',                'Point 7 displacement'),
        ('Embedded t-SNE',  'Dimension 1'),
        ('',                'Dimension 2'),
        ('',                'Dimension 3'),
        ('EM-GMM',          'Assignment No.')],
        names=['Type', 'Frame@10Hz'])
    df_training_data = pd.DataFrame(all_data_as_array, columns=multi_index_columns)
    time_str = time.strftime("_%Y%m%d_%H%M")

    # Save DataFrames to CSV
    df_training_data.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_trainlabels_10Hz_{time_str}.csv')), index=True, chunksize=10000, encoding='utf-8')

    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'wb') as f:
        joblib.dump(classifier, f)

    logger.info(f'Saved.  TODO: add specificity. Function: {inspect.stack()[0][3]}().')  # TODO: add specificity to log
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores

### RUN ###

def run_umap(predict_folders) -> Tuple[Any, Any]:
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns data_new, fs_labels: see bsoid_umap.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs.
    """
    time_str = time.strftime("_%Y%m%d_%H%M")
    # Read in model
    with open(os.path.join(OUTPUT_PATH, config.MODEL_FILENAME), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments = joblib.load(fr)

    data_new, fs_labels = classify_LEGACY.main_umap(predict_folders, VIDEO_FPS, nn_classifier)

    filenames_list = []
    all_dfs_list: List[pd.DataFrame] = []
    for idx_folder, folder in enumerate(predict_folders):  # Loop through folders
        csv_files_in_current_predict_folder: List[str] = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder)
        for idx_file, filename in enumerate(csv_files_in_current_predict_folder):
            logger.info(f'Importing CSV file {idx_file+1} from folder {idx_folder+1}')
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames_list.append(filename)
            all_dfs_list.append(curr_df)

    for i in range(len(fs_labels)):
        csv_name: str = os.path.basename(filenames_list[i]).rpartition('.')[0]
        fs_labels_pad = np.pad(fs_labels[i], (6, 0), 'edge')
        df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_dfs_list[0]]
        xyfs_df = pd.concat(frames, axis=1)
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_labels_{VIDEO_FPS}Hz_{time_str}{csv_name}.csv')), index=True, chunksize=10000, encoding='utf-8')

        runlen_df, dur_stats, df_tm = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(fs_labels[i])

        runlen_df.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_runlen_{VIDEO_FPS}Hz_{time_str}{csv_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        dur_stats.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_stats_{VIDEO_FPS}Hz_{time_str}{csv_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df_tm.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_transitions_{VIDEO_FPS}Hz_{time_str}{csv_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        if config.PLOT_GRAPHS:
            my_file = 'transition_matrix'
            # transition_matrix: np.ndarray, fps: int, save_fig_to_file=True, fig_file_prefix='transition_matrix'
            fig = util.visuals.plot_tmat(df_tm, VIDEO_FPS, save_fig_to_file=config.SAVE_GRAPHS_TO_FILE, fig_file_prefix=my_file)
            # fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, str(fps_video), 'Hz', time_str, csv_name, '.svg'))))
            file_name = f'{my_file}{VIDEO_FPS}Hz_{time_str}{csv_name}'
            util.visuals.save_graph_to_file(fig, file_name) # fig.savefig(os.path.join(OUTPUT_PATH, f'{my_file}{VIDEO_FPS}Hz_{time_str}{csv_name}.svg'))  # fig.savefig(os.path.join(OUTPUT_PATH, f'{my_file}{VIDEO_FPS}Hz_{time_str}{csv_name}.svg'))

    with open(os.path.join(OUTPUT_PATH, f'bsoid_predictions{time_str}.sav'), 'wb') as files_to_dump:
        joblib.dump([data_new, fs_labels], files_to_dump)

    logger.warning(f'{inspect.stack()[0][3]}:All saved. Expand on this message later, then downgrade to INFO or DEBUG')
    return data_new, fs_labels
def run_voc(predict_folders) -> Tuple[Any, Any, Any, Any]:
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_voc.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in fps_video, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. Neural net predicted labels (number of training data points by 1, columns 8)
    """
    model_filename = config.MODEL_FILENAME  # model_filename = f'bsoid_{config.MODEL_NAME}.sav'
    with open(os.path.join(OUTPUT_PATH, model_filename), 'rb') as fr:
        behv_model = joblib.load(fr)  # TODO: isn't this function loading up a 2-tuple?

    data_new, feats_new, labels_fslow, labels_fshigh = classify_LEGACY.main_voc(predict_folders, VIDEO_FPS, behv_model)
    filenames = []
    all_df = []
    for idx_folder, folder in enumerate(predict_folders):  # Loop through folders
        f = util.likelihoodprocessing.get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder)
        for j, filename in enumerate(f):
            logger.info(f'Importing CSV file {j+1} from folder {idx_folder+1}')
            curr_df = pd.read_csv(filename, low_memory=False)

            filenames.append(filename)
            all_df.append(curr_df)

    for i in range(len(feats_new)):
        all_data = np.concatenate([feats_new[i].T, labels_fslow[i].reshape(len(labels_fslow[i]), 1)], axis=1)
        multi_index_columns = pd.MultiIndex.from_tuples([
            ('Features', 'Distance between points 1 & 5'),
            ('', 'Distance between points 1 & 8'),
            ('', 'Angle change between points 1 & 2'),
            ('', 'Angle change between points 1 & 4'),
            ('', 'Point 3 displacement'),
            ('', 'Point 7 displacement'),
            ('Neural net classifier', 'B-SOiD labels')],
            names=['Type', 'Frame@10Hz'])
        df_predictions = pd.DataFrame(all_data, columns=multi_index_columns)
        time_str = time.strftime("_%Y%m%d_%H%M")
        csv_file_name = os.path.basename(filenames[i]).rpartition('.')[0]
        df_predictions.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_labels_10Hz_{time_str}{csv_file_name}.csv')), index=True, chunksize=10000, encoding='utf-8')

        df_runlengths_i, df_duration_stats_i, df_transition_matrix_i = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(labels_fslow[i])  # runlen_df1, dur_stats1, df_tm1 = util.statistics.main(labels_fslow[i]) # # runlen_df1, dur_stats1, df_tm1 = bsoid_voc.utils.statistics.main(labels_fslow[i])

        if config.PLOT_GRAPHS:
            util.visuals.plot_transition_matrix(df_transition_matrix_i, VIDEO_FPS, save_fig_to_file=config.SAVE_GRAPHS_TO_FILE, fig_file_prefix='transition_matrix')
        df_runlengths_i.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_runlen_10Hz_{time_str}{csv_file_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df_duration_stats_i.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_stats_10Hz_{time_str}{csv_file_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        df_transition_matrix_i.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_transitions_10Hz_{time_str}{csv_file_name}.csv')), index=True, chunksize=10000, encoding='utf-8')

        labels_fs_high_pad = np.pad(labels_fshigh[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_fs_high_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        csv_file_name = os.path.basename(filenames[i]).rpartition('.')[0]
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, f'bsoid_labels_{VIDEO_FPS}Hz_{time_str}{csv_file_name}.csv')), index=True, chunksize=10000, encoding='utf-8')
        runlen_df2, dur_stats2, df_transitionmatrix_2 = util.statistics.main(labels_fshigh[i])

        # runlen_df2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(fps_video), 'Hz', timestr, csvname, '.csv')))), index=True, chunksize=10000, encoding='utf-8')
        runlen_df2.to_csv(os.path.join(OUTPUT_PATH, f'bsoid_runlen_{VIDEO_FPS}Hz_{time_str}{csv_file_name}.csv'), index=True, chunksize=10000, encoding='utf-8')
        dur_stats2.to_csv(os.path.join(OUTPUT_PATH, f'bsoid_stats_{VIDEO_FPS}Hz_{time_str}{csv_file_name}.csv'), index=True, chunksize=10000, encoding='utf-8')
        df_transitionmatrix_2.to_csv(os.path.join(OUTPUT_PATH, f'bsoid_transitions_{VIDEO_FPS}Hz_{time_str}{csv_file_name}.csv'), index=True, chunksize=10000, encoding='utf-8')
    logger.info('All saved.')
    return data_new, feats_new, labels_fslow, labels_fshigh


### LEGACY 'main' FUNCTIONS BELOW-- POTENTIAL FUTURE DEPRECATIONS ###
"""
:param train_folders: list, folders to build behavioral model on
:param predict_folders: list, folders to run prediction using behavioral model
:returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
:returns feats_new, labels_fslow, labels_fshigh: see bsoid_py.classify
Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
Automatically saves CSV files containing training and new outputs
"""
def main_py(train_folders, predict_folders):
    deprec_warning = f'This function,' \
                     f'{util.likelihoodprocessing.get_current_function()}, is likely to be deprecated in future. ' \
                     f'It combines bsoid_py:build() and bsoid_py:run() implicitly which is not great.'
    logger.warning(deprec_warning)
    features_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores = build_py(train_folders)
    data_new, feats_new, labels_fslow, labels_fshigh = run_py(predict_folders)
    return features_10fps, trained_tsne, scaler, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh
def main_voc(train_folders, predict_folders):
    deprec_warning = f'This function,' \
                     f'{util.likelihoodprocessing.get_current_function()}, is likely to be deprecated in future.'
    logger.warning(deprec_warning)
    features_10fps, trained_tsne, gmm_assignments, classifier, scores = build_voc(train_folders)
    data_new, feats_new, labels_fslow, labels_fshigh = run_voc(predict_folders)
    return features_10fps, trained_tsne, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh
def main_umap(train_folders, predict_folders):
    deprec_warning = f'This function,' \
                     f'{util.likelihoodprocessing.get_current_function()}, is likely to be deprecated in future.'
    logger.warning(deprec_warning)
    f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
    scores, nn_assignments = build_umap(train_folders)
    data_new, fs_labels = run_umap(predict_folders)
    return f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
           scores, nn_assignments, data_new, fs_labels


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
#     vid2frame(VID_NAME, f_10fps[ID], fps_video, FRAME_DIR)
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


########################################################################################################################

@config.deco__log_entry_exit(logger)
def test_function_to_build_then_run_umap():
    logger.debug(f'STARTING _UMAP TRAIN SERIES')
    build_umap(config.TRAIN_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDED _UMAP TRAIN SERIES SUCCESSFULLY')
    logger.debug(f'STARTING _UMAP RUN SERIES')
    run_umap(config.PREDICT_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDING _UMAP RUN SERIES. SUCCESS!')
    logger.debug(f'End of test.')
@config.deco__log_entry_exit(logger)
def test_function_to_build_then_run_voc():
    logger.debug(f'STARTING _VOC TRAIN SERIES')
    build_voc(config.TRAIN_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDED _VOC TRAIN SERIES SUCCESSFULLY')
    logger.debug(f'STARTING _VOC RUN SERIES')
    run_voc(config.PREDICT_FOLDERS_IN_DLC_PROJECT)
    logger.debug(f'ENDING _VOC RUN SERIES. SUCCESS!')
    logger.debug(f'End of test.')
@config.deco__log_entry_exit(logger)
def test_build_py():
    build_py(config.TRAIN_FOLDERS_IN_DLC_PROJECT)

# if __name__ == "__main__":  # umap
#     f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
#         scores, nn_assignments, data_new, fs_labels = main_umap(TRAIN_FOLDERS, PREDICT_FOLDERS)

# if __name__ == "__main__":  # voc
#     f_10fps, trained_tsne, gmm_assignments, classifier, scores, data_new, feats_new, labels_fslow, labels_fshigh = main_voc(TRAIN_FOLDERS, PREDICT_FOLDERS)


if __name__ == "__main__":  # py
    # test_build_py()
    test_function_to_build_then_run_py()  # main_py(TRAIN_FOLDERS, PREDICT_FOLDERS)
    # test_function_to_build_then_run_umap()
