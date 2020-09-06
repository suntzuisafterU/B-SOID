"""


Every function in this file is an entire runtime sequence encapsulated.
"""
from typing import List
import inspect
import joblib
import pandas as pd
import numpy as np
import os
import time


from bsoid import config
from bsoid.util import visuals


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
        files_not_placeholder = [f for f in os.listdir(folder_path) if
                                 f != '.placeholder' and not os.path.isdir(os.path.join(folder_path, f))]
        # Loop over remaining files (within current folder iter) that are to be deleted next
        for non_placeholder_file in files_not_placeholder:
            file_to_delete_full_path = os.path.join(folder_path, non_placeholder_file)
            logger.debug(f'{inspect.stack()[0][3]}(): Deleting file: {file_to_delete_full_path}')
            os.remove(file_to_delete_full_path)

    return None


########################################################################################################################

def sample_runtime_function(*args, **kwargs):
    time.sleep(3)
    return


# def build_classifier_and_save_to_file__legacy_py():
    # """
    # Mimics the original implementation of bsoid_py/main.py:build()
    #
    #     Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
    #     1. original features (number of training data points by 7 dimensions, columns 1-7)
    #     2. embedded features (number of training data points by 3 dimensions, columns 8-10)
    #     3. em-gmm assignments (number of training data points by 1, columns 11)
    # Automatically saves classifier in OUTPUT_PATH with MODEL_NAME in LOCAL_CONFIG
    # :param train_folders: list, folders to build behavioral model on
    # :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    # """
    #
    # # Get data
    #
    # # Extract features
    #
    # # Train TSNE
    #
    # # Train GMM
    #
    # # Train SVM
    #
    #
    # # Plot as necessary
    # if config.PLOT_GRAPHS:
    #     logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
    #     visuals.plot_classes_EMGMM_assignments(trained_tsne_list, gmm_assignments, config.SAVE_GRAPHS_TO_FILE)
    #     visuals.plot_accuracy_SVM(scores)
    #     visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
    #     logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')
    # # ?
    # all_data = np.concatenate([
    #     features_10fps.T,
    #     trained_tsne,
    #     gmm_assignments.reshape(len(gmm_assignments), 1)
    # ], axis=1)
    #
    # multi_index_columns = pd.MultiIndex.from_tuples([
    #     ('Features',        'Relative snout to forepaws placement'),
    #     ('',                'Relative snout to hind paws placement'),
    #     ('',                'Inter-forepaw distance'),
    #     ('',                'Body length'),
    #     ('',                'Body angle'),
    #     ('',                'Snout displacement'),
    #     ('',                'Tail-base displacement'),
    #     ('Embedded t-SNE',  'Dimension 1'),
    #     ('',                'Dimension 2'),
    #     ('',                'Dimension 3'),
    #     ('EM-GMM',          'Assignment No.')],
    #     names=['Type', 'Frame@10Hz'])
    # df_training_data = pd.DataFrame(all_data, columns=multi_index_columns)
    #
    # # Write training data to csv
    # df_training_data.to_csv(os.path.join(config.OUTPUT_PATH, f'bsoid_trainlabels_10Hz{config.runtime_timestr}.csv'),
    #                         index=True, chunksize=10000, encoding='utf-8')
    #
    # # Save model data to file
    # with open(os.path.join(config.OUTPUT_PATH, f'bsoid_{config.MODEL_NAME}.sav'), 'wb') as model_file:
    #     joblib.dump([classifier, scaler_object], model_file)
    #
    # logger.error(f'{inspect.stack()[0][3]}: Saved stuff...elaborate on this message later.')  # TODO: elaborate on log message
    # return features_10fps, trained_tsne, scaler_object, gmm_assignments, classifier, scores
