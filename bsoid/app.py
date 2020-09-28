"""


Every function in this file is an entire runtime sequence encapsulated.
"""
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Any, List, Tuple
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import time


from bsoid import classify, classify_LEGACY, config, feature_engineering, train, train_LEGACY, util
from bsoid.config import OUTPUT_PATH, VIDEO_FPS
from bsoid.util.bsoid_logging import get_current_function  # for debugging purposes


logger = config.initialize_logger(__name__)


###

def build_classifier_new_pipeline(train_folders: List[str] = config.TRAIN_DATA_FOLDER_PATH) -> None:
    """
    new build_py implementation for new pipeline -- TEST
    1) retrieve data
    2) adaptively filter data
    3) extract features
    4) train tsne
    5) train EM/GMM
    6) Train SVM
    """
    # Arg check
    if len(train_folders) == 0:
        warn_empty_train_folders_paths = f'Empty train folders list. No train folders specified.' \
                                         f'train_folders = {train_folders} / ' \
                                         f'config.train_folders_paths = {config.TRAIN_DATA_FOLDER_PATH}'
        logger.exception(warn_empty_train_folders_paths)
        raise RuntimeError(warn_empty_train_folders_paths)

    # # # 1) Get data
    dfs_unfiltered_list: List[pd.DataFrame] = []

    # Loop over train folders to fetch data
    for train_path in config.TRAIN_FOLDERS_PATHS_toBeDeprecated:
        logger.debug(f'train_path = {train_path}')
        csv_files_paths: List[str] = util.io.check_folder_contents_for_csv_files(train_path, check_recursively=True)
        logger.debug(f'csvs = {csv_files_paths}')
        for file_csv in csv_files_paths:
            df_csv_i: pd.DataFrame = util.io.read_csv(file_csv)
            logger.debug(f'CSV read in: {file_csv}')
            dfs_unfiltered_list.append(df_csv_i)
            break  # Debugging effort. Remove this line later. todo
        break  # Debugging effort. Remove this line later. todo*
    logger.debug(f'len(dfsList) = {len(dfs_unfiltered_list)}')

    if len(dfs_unfiltered_list) == 0:
        err_zero_csvs = f'{get_current_function()}(): In the course of pulling CSVs to process in pipeline, ' \
                         f'zero CSVs were read-in. Check that TRAIN_FOLDERS_PATHS ({config.TRAIN_FOLDERS_PATHS_toBeDeprecated}) ' \
                         f'are valid spots to check for DLC csvs. '
        logger.error(err_zero_csvs)
        # For now, raise Exception just so that we can catch the bug in pathing more obviously during development.
        # Later, we can see if we want to fail silently and just rely on the logger logging the warning.
        raise RuntimeError(err_zero_csvs)

    # # # 2) Adaptively filter
    dfs_list_filtered: List[Tuple[pd.DataFrame, List[float]]] = [feature_engineering.adaptively_filter_dlc_output(df)
                                                                 for df in dfs_unfiltered_list]

    # # # 3) Engineer features
    # Loop over DataFrames, engineer features!
    dfs_engineered_features: List[List[pd.DataFrame, List[float]]] = [
        [feature_engineering.engineer_7_features_dataframe(df), i] for df, i in dfs_list_filtered]

    # Engineer features into 100ms bins
    for i, df_rect_tuple in enumerate(dfs_engineered_features):
        df_i, rect_i = df_rect_tuple
        for column in df_i.columns:
            if 'scorer' not in column:
                dfs_engineered_features[i][0][column] = \
                    feature_engineering.integrate_df_feature_into_bins(df_i, column)

    df_all_features: pd.DataFrame = pd.concat([df for df, _ in dfs_engineered_features])

    raise Exception(f'Done for now')

    #####

    features_of_interest = [

    ]
    label = 'TBD FIX ME LATER'  # ?

    # Train TSNE

    df_features_10fps, df_features_10fps_scaled, trained_tsne, scaler = train.train_TSNE_NEW(df_all_features, features_of_interest)

    # Train GMM
    df['assignments'] = train_LEGACY.train_emgmm_with_learned_tsne_space_NEW(df[features])  # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below

    # Train SVM
    classifier: object = train.train_SVM__bsoid_svm_py(df_features_10fps_scaled, features_of_interest, label)


    # Plot to view progress if necessary
    if config.PLOT_GRAPHS:
        feats_train, feats_test, labels_train, labels_test = train_test_split(df[features_of_interest], df[label])
        scores = cross_val_score(classifier, feats_test, labels_test,
                                 cv=config.CROSSVALIDATION_K, n_jobs=config.CROSSVALIDATION_N_JOBS)
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        util.visuals.plot_GM_assignments_in_3d(trained_tsne, df['assignments'], config.SAVE_GRAPHS_TO_FILE)
        util.visuals.plot_accuracy_SVM(scores)
        util.visuals.plot_feats_bsoidpy(df_features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')


    file_names_list, list_of_arrays_of_training_data, _perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)


    all_data = np.concatenate([
        df_features_10fps.T,
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

    # Write training data to csv
    training_data_labels_10hz_csv_filename = f'BSOiD__{config.MODEL_NAME}__' \
                                             f'train_data_labels__10Hz__{config.runtime_timestr}.csv'
    df_training_data.to_csv(os.path.join(OUTPUT_PATH, training_data_labels_10hz_csv_filename),
                            index=True, chunksize=10000, encoding='utf-8')

    # Save model data to file
    with open(os.path.join(OUTPUT_PATH, config.MODEL_FILENAME), 'wb') as model_file:
        joblib.dump([classifier, scaler_object], model_file)

    logger.error(f'{inspect.stack()[0][3]}: Saved model to file in the form of: [classifier, scaler_object]')  # TODO: see msg

    # END BUILD_PY
    return


def run_classifier_new_pipeline():
    """

    """
    path_to_model_file = os.path.join(OUTPUT_PATH, config.MODEL_FILENAME)
    try:
        with open(path_to_model_file, 'rb') as fr:
            behavioural_model, train_scaler_obj = joblib.load(fr)
    except FileNotFoundError as fnfe:
        file_not_found_err = f'Model not found: {path_to_model_file}.'  # TODO: HIGH: expand on err
        logger.error(file_not_found_err)
        raise FileNotFoundError(f'{file_not_found_err} // original EXCEPTION: {repr(fnfe)}.')

    # TODO: below import data BREAKS because predict folder not necessarily in DLC0
    filenames, data_new, perc_rect = util.likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(config.PREDICT_FOLDERS_IN_DLC_PROJECT_toBeDeprecated)
    ### Extract features
    # features_new = classify.bsoid_extract_py(data_new)
    intermediate_features = feature_engineering.extract_7_features_bsoid_tsne_py(data_new)  # REPLACEMENT FOR ABOVE
    features_new = feature_engineering.integrate_features_into_100ms_bins_LEGACY(data=data_new, features=intermediate_features, fps=config.VIDEO_FPS)

    df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe()
    feature_engineering.integrate_df_feature_into_bins()

    # Predict labels
    labels_frameshift_low: List = classify.bsoid_predict_py(features_new, train_scaler_obj, behavioural_model)
    # Create
    labels_frameshift_high: List = classify_LEGACY.bsoid_frameshift_py(data_new, train_scaler_obj, config.VIDEO_FPS, behavioural_model)

    if config.PLOT_GRAPHS:
        util.visuals.plot_feats_bsoidpy(features_new, labels_frameshift_low)

    # TODO: HIGH: Ensure that the labels predicted on predict_folders matches to the video that will be labeled hereafter
    if config.GENERATE_VIDEOS:
        if len(labels_frameshift_low) > 0:
            # 1/2 write frames to disk
            util.videoprocessing.write_annotated_frames_to_disk_from_video(
                config.VIDEO_TO_LABEL_PATH,
                labels_frameshift_low[config.IDENTIFICATION_ORDER]
            )
            # 2/2 created labeled video
            util.videoprocessing.create_labeled_vid(
                labels_frameshift_low[config.IDENTIFICATION_ORDER],
                critical_behaviour_minimum_duration=3,
                num_randomly_generated_examples=5,
                frame_dir=config.FRAMES_OUTPUT_PATH,
                output_path=config.SHORT_VIDEOS_OUTPUT_PATH
            )
        else:
            logger.error(f'{inspect.stack()[0][3]}(): config.GENERATE_VIDEOS = {config.GENERATE_VIDEOS}; '
                         f'however, the generation of '
                         f'a video could NOT occur because labels_fs_low is a list of length zero and '
                         f'config.ID is attempting to index an empty list.')

    return data_new, features_new, labels_frameshift_low, labels_frameshift_high

    data_new, features_new, labels_fs_low, labels_fs_high = classify.main_py(predict_folders, train_scaler_obj, VIDEO_FPS, behavioural_model)


    filenames: List[str] = []
    all_dfs_list: List[pd.DataFrame] = []
    for i, folder in enumerate(predict_folders):  # Loop through folders
        file_names_csvs: List[
            str] = io.get_filenames_csvs_from_folders_recursively_in_dlc_project_path(folder)
        for j, csv_filename in enumerate(file_names_csvs):
            logger.info(f'{inspect.stack()[0][3]}(): Importing CSV file {j + 1} from folder {i + 1}.')
            curr_df = pd.read_csv(csv_filename, low_memory=False)
            filenames.append(csv_filename)
            all_dfs_list.append(curr_df)

    for i, feature_new_i in enumerate(features_new):
        all_data: np.ndarray = np.concatenate([
            feature_new_i.T,
            labels_fs_low[i].reshape(len(labels_fs_low[i]), 1),
        ], axis=1)
        multi_index_columns = pd.MultiIndex.from_tuples([
            ('Features', 'Relative snout to forepaws placement'),
            ('', 'Relative snout to hind paws placement'),
            ('', 'Inter-forepaw distance'),
            ('', 'Body length'),
            ('', 'Body angle'),
            ('', 'Snout displacement'),
            ('', 'Tail-base displacement'),
            ('SVM classifier', 'B-SOiD labels')],
            names=['Type', 'Frame@10Hz'])
        df_predictions = pd.DataFrame(all_data, columns=multi_index_columns)

        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        df_predictions.to_csv(
            (os.path.join(OUTPUT_PATH, f'BSOiD__labels__10Hz__{config.runtime_timestr}__{csvname}.csv')), index=True,
            chunksize=10000, encoding='utf-8')

        runlen_df1, dur_stats1, df_tm1 = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(
            labels_fs_low[i])

        # if PLOT_TRAINING:
        #     plot_tmat(df_tm1, fps_video)

        # Save (stuff?) to CSV

        runlen_df1.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__runlengths__10Hz__{config.runtime_timestr}__{csvname}.csv')),
                          index=True, chunksize=10000, encoding='utf-8')
        dur_stats1.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__statistics__10Hz__{config.runtime_timestr}__{csvname}.csv')),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm1.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__transitions__10Hz__{config.runtime_timestr}__{csvname}.csv')), index=True,
                      chunksize=10000, encoding='utf-8')

        #
        labels_fshigh_pad = np.pad(labels_fs_high[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_fshigh_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''  # TODO: low: address duplicate line here and below
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_dfs_list[0]]
        df_xy_fs = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]

        # runlen_df2.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(VIDEO_FPS), 'Hz', timestr, csvname,
        df_xy_fs.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__labels__{VIDEO_FPS}Hz{time_str}__{csvname}.csv')),
                        index=True, chunksize=10000, encoding='utf-8')

        runlen_df2, dur_stats2, df_tm2 = util.statistics.get_runlengths_statistics_transition_matrix_from_labels(
            labels_fs_high[i])
        runlen_df2.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__NeedsAName__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')),
                          index=True, chunksize=10000, encoding='utf-8')

        # TODO: ############### Reformat the below lines using f-strings #################################################################################
        dur_stats2.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__statistics__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')),
                          index=True, chunksize=10000, encoding='utf-8')
        df_tm2.to_csv((os.path.join(OUTPUT_PATH, f'BSOiD__transitions__{VIDEO_FPS}Hz__{time_str}__{csvname}.csv')),
                      index=True, chunksize=10000, encoding='utf-8')

    #
    with open(os.path.join(OUTPUT_PATH, 'bsoid_predictions.sav'), 'wb') as f:
        joblib.dump([labels_fs_low, labels_fs_high], f)

    logger.info('All saved. Expand on this info message later.')
    return data_new, features_new, labels_fs_low, labels_fs_high


def clear_output_folders() -> None:
    """
    For each folder specified below (magic variables be damned),
        delete everything in that folder except for the .placeholder file and any sub-folders there-in.
    """
    # Choose folders to clear (currently set as magic variables in function below)
    folders_to_clear: List[str] = [config.OUTPUT_PATH, config.GRAPH_OUTPUT_PATH,
                                   config.SHORT_VIDEOS_OUTPUT_PATH, config.FRAMES_OUTPUT_PATH]
    # Loop over all folders to empty
    for folder_path in folders_to_clear:
        # Parse all files in current folder_path, but exclusive placeholders, folders
        valid_files_to_delete = [file_name for file_name in os.listdir(folder_path)
                                 if file_name != '.placeholder'
                                 and not os.path.isdir(os.path.join(folder_path, file_name))]
        # Loop over remaining files (within current folder iter) that are to be deleted next
        for file in valid_files_to_delete:
            file_to_delete_full_path = os.path.join(folder_path, file)
            try:
                os.remove(file_to_delete_full_path)
                logger.debug(f'{inspect.stack()[0][3]}(): Deleted file: {file_to_delete_full_path}')
            except PermissionError as pe:
                logger.warning(f'{inspect.stack()[0][3]}(): Could not delete file: {file_to_delete_full_path} / '
                               f'{repr(pe)}')

    return None


def clear_logs() -> None:
    log_folder = config.config_file_log_folder_path
    confirmation = input(f'You are about to destroy ALL files in your log '
                         f'folder ({log_folder}) -- are you sure you want to do this? [Y/N]: ').strip().upper()

    if confirmation == 'Y' or confirmation == 'YES':
        files_to_delete = [file for file in os.listdir(log_folder) if file != '.placeholder']
        for file in files_to_delete:
            file_to_delete_path = os.path.join(log_folder, file)
            try:
                os.remove(file_to_delete_path)
                logger.debug(f'{inspect.stack()[0][3]}(): deleted file: {file_to_delete_path}')
            except PermissionError as pe:
                logger.warning(f'{inspect.stack()[0][3]}(): Could not delete file: {file_to_delete_path} / {repr(pe)}')
    else:
        print('Clearing log files canceled. Have a great day!')

    return


def TEST_readcsv():
    # A test function to see if breakpoints work on read_csv
    path = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\' \
           f'sample_train_data_folder\\Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
    util.io.read_csv(path)


def build_and_run_new_pipeline():
    build_classifier_new_pipeline()
    run_classifier_new_pipeline()
    return


########################################################################################################################

def sample_runtime_function(*args, **kwargs):
    time.sleep(3)
    return


########################################################################################################################

if __name__ == '__main__':
    # Run this file for ebugging purposes
    build_classifier_new_pipeline()
    # TEST_readcsv()

