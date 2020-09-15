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


def build_classifier_new_pipeline() -> None:
    """

    """
    # Get data
    dfs_raw_list: List[pd.DataFrame] = []
    for train_path in config.TRAIN_FOLDERS_PATHS:
        logger.debug(f'train_path = {train_path}')
        csvs: List[str] = io.check_folder_contents_for_csv_files(train_path, check_recursively=True)
        logger.debug(f'csvs = {csvs}')
        for csv_i in csvs:
            df_csv_i: pd.DataFrame = io.read_csv(csv_i)
            dfs_raw_list.append(df_csv_i)
    logger.debug(f'len(dfsList) = {len(dfs_raw_list)}')

    # Adaptively filter
    # TODO: finish adaptive filtering function
    dfs_list_filtered: List[Tuple[pd.DataFrame, List]] = [feature_engineering.adaptively_filter_dlc_output(df)
                                                          for df in dfs_raw_list]


    # Loop over DataFrames, engineer features!
    # TODO: finish feature engineering
    dfs_engineered_features: List[pd.DataFrame] = [feature_engineering.extract_features(df)
                                                   for df, _ in dfs_list_filtered]





    ####################################################################################################################


    file_names_list, list_of_arrays_of_training_data, _perc_rect = likelihoodprocessing.import_csvs_data_from_folders_in_PROJECTPATH_and_process_data(train_folders)




    # Check that outputs are fine for runtime
    if len(file_names_list) == 0:
        zero_folders_error = f'{inspect.stack()[0][3]}: Zero training folders were specified. Check ' \
                             f'your config file!!! Train folders = {train_folders} // Filenames = {file_names_list}.'
        logger.error(zero_folders_error)
        raise ValueError(zero_folders_error)
    if len(file_names_list[0]) == 0:
        zero_filenames_error = f'{inspect.stack()[0][3]}: Zero file names were found. filenames = {file_names_list}.'
        logger.error(zero_filenames_error)
        raise ValueError(zero_filenames_error)





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


    return features_10fps, trained_tsne_list, scaler, gmm_assignments, classifier, scores

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

