"""


Every function in this file is an entire runtime sequence encapsulated.
"""
from typing import Any, List, Tuple
import inspect
import pandas as pd
import os
import time


from bsoid import config, feature_engineering
from bsoid.util import io, visuals
from bsoid.util.bsoid_logging import get_current_function  # for debugging purposes


logger = config.initialize_logger(__name__)


###

def TEST_readcsv():
    # A test function to see if breakpoints work on read_csv
    path = f'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\EPM-DLC-projects\\' \
           f'sample_train_data_folder\\Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
    io.read_csv(path)


def build_classifier_new_pipeline(train_folders: List[str] = config.TRAIN_FOLDERS_PATHS) -> None:
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
                                         f'config.train_folders_paths = {config.TRAIN_FOLDERS_PATHS}'
        logger.exception(warn_empty_train_folders_paths)
        raise RuntimeError(warn_empty_train_folders_paths)

    # # # 1) Get data
    dfs_unfiltered_list: List[pd.DataFrame] = []

    for train_path in config.TRAIN_FOLDERS_PATHS:
        logger.debug(f'train_path = {train_path}')
        csv_files_paths: List[str] = io.check_folder_contents_for_csv_files(train_path, check_recursively=True)
        logger.debug(f'csvs = {csv_files_paths}')
        for file_csv in csv_files_paths:
            df_csv_i: pd.DataFrame = io.read_csv(file_csv)
            logger.debug(f'CSV read in: {file_csv}')
            dfs_unfiltered_list.append(df_csv_i)
            break  # Debugging effort
        break  # Debugging effort
    logger.debug(f'len(dfsList) = {len(dfs_unfiltered_list)}')

    if len(dfs_unfiltered_list) == 0:
        warn_zero_csvs = f'{get_current_function()}(): In the course of pulling CSVs to process in pipeline, ' \
                         f'zero CSVs were read-in. Check that TRAIN_FOLDERS_PATHS ({config.TRAIN_FOLDERS_PATHS}) ' \
                         f'are valid spots to check for DLC csvs. '
        logger.error(warn_zero_csvs)
        # For now, raise Exception just so that we can catch the bug in pathing more obviously during development.
        # Later, we can see if we want to fail silently and just rely on the logger logging the warning.
        raise RuntimeError(warn_zero_csvs)

    # # # 2) Adaptively filter
    # TODO: finish adaptive filtering function
    dfs_list_filtered: List[Tuple[pd.DataFrame, List]] = [feature_engineering.adaptively_filter_dlc_output(df)
                                                          for df in dfs_unfiltered_list]

    # Loop over DataFrames, engineer features!
    # TODO: finish feature engineering
    dfs_engineered_features: List[pd.DataFrame] = [feature_engineering.engineer_7_features_dataframe(df)
                                                   for df, _ in dfs_list_filtered]



    raise Exception(f'Done for now')


    ####################################################################################################################


    file_names_list, list_of_arrays_of_training_data, _perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)


    # Train TSNE
    features_10fps, features_10fps_scaled, trained_tsne_list, scaler = extract_features_and_train_TSNE(list_of_arrays_of_training_data)  # features_10fps, features_10fps_scaled, trained_tsne_list, scaler = bsoid_tsne_py(list_of_arrays_of_training_data)  # replace with: extract_features_and_train_TSNE

    # Train GMM
    gmm_assignments = train_emgmm_with_learned_tsne_space(trained_tsne_list)  # gmm_assignments = bsoid_gmm_pyvoc(trained_tsne)  # replaced with below

    # Train SVM
    classifier, scores = bsoid_svm_py(features_10fps_scaled, gmm_assignments)

    # Plot to view progress if necessary
    if config.PLOT_GRAPHS:
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        visuals.plot_classes_EMGMM_assignments(trained_tsne_list, gmm_assignments, config.SAVE_GRAPHS_TO_FILE)
        visuals.plot_accuracy_SVM(scores)
        visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')



    # return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores
    # features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores = train.get_data_train_TSNE_then_GMM_then_SVM_then_return_EVERYTHING__py(train_folders)

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

    # Write training data to csv
    training_data_labels_10hz_csv_filename = f'BSOiD__{config.MODEL_NAME}__' \
                                             f'train_data_labels__10Hz__{config.runtime_timestr}.csv'
    df_training_data.to_csv(os.path.join(OUTPUT_PATH, training_data_labels_10hz_csv_filename),
                            index=True, chunksize=10000, encoding='utf-8')

    # Save model data to file
    with open(os.path.join(OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'wb') as model_file:
        joblib.dump([classifier, scaler_object], model_file)

    logger.error(f'{inspect.stack()[0][3]}: Saved model to file. Form: [classifier, scaler_object]')  # TODO: see msg
    return features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores


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


########################################################################################################################

def sample_runtime_function(*args, **kwargs):
    time.sleep(3)
    return


# def build_classifier_and_save_to_file__legacy_py():
#     """
#     Mimics the original implementation of bsoid_py/main.py:build()
#
#         Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
#         1. original features (number of training data points by 7 dimensions, columns 1-7)
#         2. embedded features (number of training data points by 3 dimensions, columns 8-10)
#         3. em-gmm assignments (number of training data points by 1, columns 11)
#     Automatically saves classifier in OUTPUT_PATH with MODEL_NAME in LOCAL_CONFIG
#     :param train_folders: list, folders to build behavioral model on
#     :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
#     """
#
#     # Get data
#
#     # Extract features
#
#     # Train TSNE
#
#     # Train GMM
#
#     # Train SVM
#
#
#     # Plot as necessary
#     if config.PLOT_GRAPHS:
#         logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
#         visuals.plot_classes_EMGMM_assignments(trained_tsne_list, gmm_assignments, config.SAVE_GRAPHS_TO_FILE)
#         visuals.plot_accuracy_SVM(scores)
#         visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
#         logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')
#     # ?
#     all_data = np.concatenate([
#         features_10fps.T,
#         trained_tsne,
#         gmm_assignments.reshape(len(gmm_assignments), 1)
#     ], axis=1)
#
#     multi_index_columns = pd.MultiIndex.from_tuples([
#         ('Features',        'Relative snout to forepaws placement'),
#         ('',                'Relative snout to hind paws placement'),
#         ('',                'Inter-forepaw distance'),
#         ('',                'Body length'),
#         ('',                'Body angle'),
#         ('',                'Snout displacement'),
#         ('',                'Tail-base displacement'),
#         ('Embedded t-SNE',  'Dimension 1'),
#         ('',                'Dimension 2'),
#         ('',                'Dimension 3'),
#         ('EM-GMM',          'Assignment No.')],
#         names=['Type', 'Frame@10Hz'])
#     df_training_data = pd.DataFrame(all_data, columns=multi_index_columns)
#
#     # Write training data to csv
#     df_training_data.to_csv(os.path.join(config.OUTPUT_PATH, f'bsoid_trainlabels_10Hz{config.runtime_timestr}.csv'),
#                             index=True, chunksize=10000, encoding='utf-8')
#
#     # Save model data to file
#     with open(os.path.join(config.OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'wb') as model_file:
#         joblib.dump([classifier, scaler_object], model_file)
#
#     logger.error(f'{inspect.stack()[0][3]}: Saved stuff...elaborate on this message later.')  # TODO: elaborate on log message
#     return features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores


if __name__ == '__main__':
    # Run this file for ebugging purposes
    build_classifier_new_pipeline()
    # TEST_readcsv()

