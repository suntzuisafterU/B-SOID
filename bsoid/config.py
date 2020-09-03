"""
Set up runtime configuration here.

*** Do not set up any variables in this file by hand. If you need to instantiate new variables,
    write them into the config.ini file, then parse with the ConfigParser ***


If you want to use this file to instantiate your config options (because, for example,
pathing may be easier), then leave the config file value blank but do not delete the key.

Reading the key/value pair from the config object requires you to first index the section then index the key.
    e.g.: input: config['SectionOfInterest']['KeyOfInterest'] -> output: 'value of interest'
Another way to od it is using the object method:
    e.g.: input: config.get('section', 'key') -> output: 'value of interest'
All values read from the config.ini file are string so type conversion must be made for non-string information.
"""

from ast import literal_eval
from pathlib import Path
from typing import List
import configparser
import logging
import os
import random
import time

from bsoid.util import logger_config

cfig_log_entry_exit: callable = logger_config.log_entry_exit  # TODO: temporary measure to enable logging when entering/exiting functions

########################################################################################################################
# Fetch the B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory to where you want the analysis to be stored
default_output_path = os.path.join(BSOID_BASE_PROJECT_PATH, 'output')
# Set runtime string for consistency
runtime_timestr = time.strftime("_%Y%m%d_%H%M")
# Set loggers default vars
default_log_folder_path = Path(BSOID_BASE_PROJECT_PATH, 'logs').absolute()
default_log_file_name = 'default.log'
# set default config file name
config_file_name = 'config.ini'
# Load up config file
configuration = configparser.ConfigParser()
configuration.read(os.path.join(BSOID_BASE_PROJECT_PATH, config_file_name))


########################################################################################################################
##### READ CONFIG FOR RUNTIME VARIABLES #####

DLC_PROJECT_PATH = configuration.get('PATH', 'DLC_PROJECT_PATH')
# Resolve output path
config_output_path = configuration.get('PATH', 'OUTPUT_PATH')
OUTPUT_PATH = config_output_path if config_output_path else default_output_path
# Resolve runtime application settings
MODEL_NAME = configuration.get('APP', 'OUTPUT_MODEL_NAME')  # Machine learning model name
RANDOM_STATE: int = configuration.getint('MODEL', 'RANDOM_STATE', fallback=random.randint(1, 100_000_000))
HOLDOUT_PERCENT: float = configuration.getfloat('MODEL', 'HOLDOUT_TEST_PCT')
CROSSVALIDATION_K: int = configuration.getint('MODEL', 'CROSS_VALIDATION_K')  # Number of iterations for cross-validation to show it's not over-fitting.
CROSSVALIDATION_N_JOBS: int = configuration.getint('MODEL', 'CROSS_VALIDATION_N_JOBS')
VIDEO_FPS: int = configuration.getint('APP', 'VIDEO_FRAME_RATE')
COMPILE_CSVS_FOR_TRAINING: int = configuration.getint('APP', 'COMPILE_CSVS_FOR_TRAINING')  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.
IDENTIFICATION_ORDER: int = configuration.getint('APP', 'FILE_IDENTIFICATION_ORDER_LEGACY')  # TODO: low: assess whether we can remove this from module altogether.
PLOT_GRAPHS: bool = configuration.getboolean('APP', 'PLOT_GRAPHS')
SAVE_GRAPHS_TO_FILE: bool = configuration.getboolean('APP', 'SAVE_GRAPHS_TO_FILE')
GENERATE_VIDEOS: bool = configuration.getboolean('APP', 'GENERATE_VIDEOS')


# Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS
# VID_NAME = os.path.join(OST_BASE_PROJECT_PATH, 'GUI_projects', 'labelled_videos', '002_ratA_inc2_above.mp4')  # '/home/aaron/Documents/OST-with-DLC/GUI_projects/labelled_videos/002_ratA_inc2_above.mp4'
VIDEO_TO_LABEL_PATH: str = configuration.get('APP', 'VIDEO_TO_LABEL_PATH')
short_video_output_directory = os.path.join(OUTPUT_PATH, 'short_videos')
assert os.path.isdir(short_video_output_directory), f'`short_video_output_directory` dir. (value={short_video_output_directory}) must exist for runtime but does not.'


SHORTVID_DIR = short_video_output_directory  # LEGACY. To be deprecated.
# ID = identification_order  # TODO: DEPRECATE. ID WAS A MISTAKE, BUT NOT SURE WHY/WHAT IT DOES


assert os.path.isdir(DLC_PROJECT_PATH), f'BASEPATH DOES NOT EXIST: {DLC_PROJECT_PATH}'
assert os.path.isdir(OUTPUT_PATH), f'OUTPUT PATH DOES NOT EXIST: {OUTPUT_PATH}'
assert os.path.isfile(VIDEO_TO_LABEL_PATH) or not VIDEO_TO_LABEL_PATH, \
    f'Video does not exist: {VIDEO_TO_LABEL_PATH}. Check pathing in config.ini file.'


# Specify where the OST project lives. Modify on your local machine as necessary.
# OST_BASE_PROJECT_PATH = configuration.get('PATH', 'OST_BASE_PROJECT_PATH')
# OST_BASE_PROJECT_PATH = os.path.join('C:\\', 'Users', 'killian', 'projects', 'OST-with-DLC')
# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'


########################################################################################################################

##### TRAIN_FOLDERS, PREDICT_FOLDERS
# TRAIN_FOLDERS, PREDICT_FOLDERS are lists of folders that are implicitly understood to exist within BASE_PATH

# Data folders used to training neural network.
# TRAIN_FOLDERS: List[str] = ['NOT_DLC_OUTPUT__SAMPLE_WITHOUT_INDEX', ]  # TRAIN_FOLDERS = [os.path.sep+'training-datasets', ]
TRAIN_FOLDERS = [
    'sample_train_data_folder',
]
for folder in TRAIN_FOLDERS:
    compiled_folder_path = os.path.join(DLC_PROJECT_PATH, folder)
    assert os.path.isdir(compiled_folder_path), f'Training folder does not exist: {compiled_folder_path}'


PREDICT_FOLDERS: List[str] = [
    'sample_predic_data_folder',
]
for folder in PREDICT_FOLDERS:
    compiled_folder_path = os.path.join(DLC_PROJECT_PATH, folder)
    assert os.path.isdir(compiled_folder_path), f'Prediction folder does not exist: {compiled_folder_path}'

# Create a folder to store extracted images.
config_value_alternate_output_path_for_annotated_frames = configuration.get(
    'PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES')

config_value_alternate_output_path_for_annotated_frames = config_value_alternate_output_path_for_annotated_frames \
    if config_value_alternate_output_path_for_annotated_frames \
    else os.path.join(OUTPUT_PATH, 'frames')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'
assert os.path.isdir(config_value_alternate_output_path_for_annotated_frames), \
    f'config_value_alternate_output_path_for_annotated_frames does not exist. ' \
    f'config_value_alternate_output_path_for_annotated_frames = ' \
    f'\'{config_value_alternate_output_path_for_annotated_frames}\'. Check config.ini pathing.'

FRAME_DIR = config_value_alternate_output_path_for_annotated_frames  # Legacy name

########################################################################################################################
##### LOGGER INSTANTIATION #####

# logging.info(f"configuration.get('LOGGING', 'LOG_FILE_NAME'):::{configuration.get('LOGGING', 'LOG_FILE_NAME')}")
# if debug == 2: print('OST PATH:::', configuration.get('PATH', 'OSTPATH', fallback=None))

# Resolve logger variables
config_file_log_folder_path = configuration.get('LOGGING', 'LOG_FILE_FOLDER_PATH')
config_file_log_folder_path = config_file_log_folder_path if config_file_log_folder_path else default_log_folder_path
# if debug >= 2: print('config_file_log_folder_path:::', config_file_log_folder_path)

config_file_name = configuration.get('LOGGING', 'LOG_FILE_NAME', fallback=default_log_file_name)
#  debug >= 2: print('config_file_name:::', config_file_name)

# Get logger variables
logger_name = configuration.get('LOGGING', 'DEFAULT_LOGGER_NAME')
log_format = configuration.get('LOGGING', 'LOG_FORMAT', raw=True)
stdout_log_level = configuration.get('LOGGING', 'STREAM_LOG_LEVEL', fallback=None)
file_log_level = configuration.get('LOGGING', 'FILE_LOG_LEVEL', fallback=None)
log_file_file_path = str(Path(config_file_log_folder_path, config_file_name).absolute())
# if debug >= 2: print('log_file_file_path AKA os.path.join(config_file_log_folder_path, config_file_name):::', log_file_file_path)
assert os.path.isdir(config_file_log_folder_path), f'Path does not exist: {config_file_log_folder_path}'


# Instantiate logger
# bsoid_logger: logging.Logger = logger_config.create_generic_logger(
#     logger_name=logger_name,
#     log_format=log_format,
#     stdout_log_level=stdout_log_level,
#     file_log_level=file_log_level,
#     file_log_file_path=log_file_file_path,
# )

# Instantiate logger decorator capable for
initialize_logger: callable = logger_config.preload_logger_with_config_vars(
    logger_name, log_format, stdout_log_level, file_log_level, log_file_file_path)


########################################################################################################################
##### MODEL PARAMETERS #####

UMAP_PARAMS = {
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=RANDOM_STATE),
}

# HDBSCAN params, density based clustering
HDBSCAN_PARAMS = {
    'min_samples': configuration.getint('HDBSCAN', 'min_samples'),  # small value
}

# EM_GMM parameters
EMGMM_PARAMS = {
    'n_components': configuration.getint('EM/GMM', 'n_components'),
    'covariance_type': configuration.get('EM/GMM', 'covariance_type'),
    'tol': configuration.getfloat('EM/GMM', 'tol'),
    'reg_covar': configuration.getfloat('EM/GMM', 'reg_covar'),
    'max_iter': configuration.getint('EM/GMM', 'max_iter'),
    'n_init': configuration.getint('EM/GMM', 'n_init'),
    'init_params': configuration.get('EM/GMM', 'init_params'),
    'verbose': configuration.getint('EM/GMM', 'verbose'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=RANDOM_STATE),
}

# Feedforward neural network (MLP) params
MLP_PARAMS = {
    'hidden_layer_sizes': literal_eval(configuration.get('MLP', 'hidden_layer_sizes')),
    'activation': configuration.get('MLP', 'activation'),
    'solver': configuration.get('MLP', 'solver'),
    'learning_rate': configuration.get('MLP', 'learning_rate'),
    'learning_rate_init': configuration.getfloat('MLP', 'learning_rate_init'),
    'alpha': configuration.getfloat('MLP', 'alpha'),
    'max_iter': configuration.getint('MLP', 'max_iter'),
    'early_stopping': configuration.getboolean('MLP', 'early_stopping'),
    'verbose': configuration.getint('MLP', 'verbose'),
}

# Multi-class support vector machine classifier params
SVM_PARAMS = {
    'C': configuration.getfloat('SVM', 'C'),
    'gamma': configuration.getfloat('SVM', 'gamma'),
    'probability': configuration.getboolean('SVM', 'probability'),
    'verbose': configuration.getint('SVM', 'verbose'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=RANDOM_STATE),
}


########################################################################################################################
##### BSOID VOC #####
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)
TSNE_PARAMS = {
    'n_components': configuration.getint('TSNE', 'n_components'),
    'n_jobs': configuration.getint('TSNE', 'n_jobs'),
    'verbose': configuration.getint('TSNE', 'verbose'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE'),
}

# TODO: HIGH: after all config parsing done, write checks that will run ON IMPORT to ensure folders exist :)


########################################################################################################################
##### LEGACY VARIABLES #####
# This version requires the six body parts Snout/Head, Forepaws/Shoulders, Hindpaws/Hips, Tailbase.
BODYPARTS_PY_LEGACY = {
    'Snout/Head': 0,
    'Neck': None,
    'Forepaw/Shoulder1': 1,
    'Forepaw/Shoulder2': 2,
    'Bodycenter': None,
    'Hindpaw/Hip1': 3,
    'Hindpaw/Hip2': 4,
    'Tailbase': 5,
    'Tailroot': None,
}

### BSOID VOC ###
# # Order the points that are encircling the mouth.
BODYPARTS_VOC_LEGACY = {
    'Point1': 0,
    'Point2': 1,
    'Point3': 2,
    'Point4': 3,
    'Point5': 4,
    'Point6': 5,
    'Point7': 6,
    'Point8': 7,
}

## *NOTE*: BASE_PATH: is likely to be deprecated in the future
# BASE_PATH = 'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\OST-DLC-projects\\pwd-may11-2020-john-howland-2020-05-11'  # TODO: HIGH: bad!!!! magic variable
# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'

if __name__ == '__main__':
    print('OUTPUTPATH:', OUTPUT_PATH)
    pass
