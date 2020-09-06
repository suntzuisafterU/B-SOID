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
import numpy as np
import os
import pandas as pd
import random
import sys
import time

# Debug opts
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=1_000)


from bsoid.util import bsoid_logging

deco__log_entry_exit: callable = bsoid_logging.log_entry_exit  # TODO: temporary measure to enable logging when entering/exiting functions

########################################################################################################################
# Fetch the B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory to where you want the analysis to be stored
default_output_path = os.path.join(BSOID_BASE_PROJECT_PATH, 'output')
# Set runtime string for consistency
runtime_timestr = time.strftime("%Y-%m-%d_%HH%MM")
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
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'graphs')
FRAMES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'frames')
SHORT_VIDEOS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'short_videos')
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
DEFAULT_SAVED_GRAPH_FILE_FORMAT: str = configuration.get('APP', 'DEFAULT_SAVED_GRAPH_FILE_FORMAT')
GENERATE_VIDEOS: bool = configuration.getboolean('APP', 'GENERATE_VIDEOS')
PERCENT_FRAMES_TO_LABEL: float = configuration.getfloat('APP', 'PERCENT_FRAMES_TO_LABEL')


# Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS  # TODO: ************* This note from the original author implies that VID_NAME must be a video that corresponds to a csv from PREDICT_FOLDERS
# VID_NAME = os.path.join(OST_BASE_PROJECT_PATH, 'GUI_projects', 'labelled_videos', '002_ratA_inc2_above.mp4')  # '/home/aaron/Documents/OST-with-DLC/GUI_projects/labelled_videos/002_ratA_inc2_above.mp4'
VIDEO_TO_LABEL_PATH: str = configuration.get('APP', 'VIDEO_TO_LABEL_PATH')
short_video_output_directory = os.path.join(OUTPUT_PATH, 'short_videos')
assert os.path.isdir(short_video_output_directory), f'`short_video_output_directory` dir. (value={short_video_output_directory}) must exist for runtime but does not.'


SHORTVID_DIR = short_video_output_directory  # LEGACY. To be deprecated.
# ID = identification_order  # TODO: DEPRECATE. ID WAS A MISTAKE, BUT NOT SURE WHY/WHAT IT DOES


assert os.path.isdir(DLC_PROJECT_PATH), f'BASEPATH DOES NOT EXIST: {DLC_PROJECT_PATH}'
assert os.path.isfile(VIDEO_TO_LABEL_PATH) or not VIDEO_TO_LABEL_PATH, \
    f'Video does not exist: {VIDEO_TO_LABEL_PATH}. Check pathing in config.ini file.'

#########################################
### OUTPUT PATH CHECK/CREATION

# TODO: create OUTPUT PATH files in case user does not use default output path in BSOID project
assert os.path.isdir(OUTPUT_PATH), f'OUTPUT PATH DOES NOT EXIST: {OUTPUT_PATH}'


########################################################################################################################

# Specify where the OST project lives. Modify on your local machine as necessary.
# OST_BASE_PROJECT_PATH = configuration.get('PATH', 'OST_BASE_PROJECT_PATH')
# OST_BASE_PROJECT_PATH = os.path.join('C:\\', 'Users', 'killian', 'projects', 'OST-with-DLC')
# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'


########################################################################################################################

### DLC PROJECT CONFIG VARS  # TODO: HIGH: IMPLEMENT
# DLC_PROJECT_CONFIG_YAML = configparser.ConfigParser()
# configuration.read(os.path.join(DLC_PROJECT_PATH, 'config.yaml'))

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
config_value_alternate_output_path_for_annotated_frames = configuration.get('PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES')

FRAMES_OUTPUT_PATH = config_value_alternate_output_path_for_annotated_frames = configuration.get('PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES') if configuration.get('PATH', 'ALTERNATE_OUTPUT_PATH_FOR_ANNOTATED_FRAMES') else FRAMES_OUTPUT_PATH  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'
assert os.path.isdir(config_value_alternate_output_path_for_annotated_frames), \
    f'config_value_alternate_output_path_for_annotated_frames does not exist. ' \
    f'config_value_alternate_output_path_for_annotated_frames = ' \
    f'\'{config_value_alternate_output_path_for_annotated_frames}\'. Check config.ini pathing.'

# FRAME_DIR = config_value_alternate_output_path_for_annotated_frames  # Legacy name

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
initialize_logger: callable = bsoid_logging.preload_logger_with_config_vars(
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
##### TESTING VARIABLES #####
max_rows_to_read_in_from_csv: int = configuration.getint('TESTING', 'max_rows_to_read_in_from_csv') if configuration.getint('TESTING', 'max_rows_to_read_in_from_csv') else sys.maxsize


########################################################################################################################
##### LEGACY VARIABLES #####
# This version requires the six body parts Snout/Head, Forepaws/Shoulders, Hindpaws/Hips, Tailbase.
BODYPARTS_PY_LEGACY = {  # It appears as though the names correlate to the expected index of the feature when in Numpy array form
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

# BSOID VOC
# # original authors' note: Order the points that are encircling the mouth.
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


###
assert isinstance(VIDEO_FPS, int), f''


def get_config_str() -> str:
    config_string = ''
    for section in configuration.sections():
        config_string += f'SECTION: {section} // OPTIONS: {configuration.options(section)}\n'
    return config_string.strip()




map_group_to_behaviour = {
    1: 'orient right',
    2: 'body lick',
    3: 'rearing',
    4: 'nose poke',
    5: 'tall wall-rear',
    6: 'face groom',
    7: 'wall-rear',
    8: 'head groom',
    9: 'nose poke',
    10: 'pause',
    11: 'locomote',
    12: 'orient right',
    13: 'paw groom',
    14: 'locomote',
    15: 'orient left',
    16: 'orient left',
}
"""


## Here are the example groups that we have extracted from multiple animals using B-SOiD

### Groups 1-4:
#### Group 1 (top left): Oreint right (+); Group 2 (top right): Body lick; 

![Mouse Action Groups 1-4](../examples/group1_4.gif)
#### Group3 (bottom left): Rearing; Group 4 (bottom right): Nose poke (+).

### Groups 5-8:
#### Group 5 (top left): Tall wall-rear; Group 6 (top right): Face groom; 
![Mouse Action Groups 5-8](../examples/group5_8.gif)
#### Group 7 (bottom left): Wall-rear; Group 8 (bottom right): Head groom.

### Groups 9-12:
#### Group 9 (top left): Nose poke (-); Group 10 (top right): Pause; 
![Mouse Action Groups 9-12](../examples/group9_12.gif)
#### Group 11 (bottom left): Locomote (+); Group 12 (bottom right): Orient right (-).

### Groups 13-16:
#### Group 13 (top left): Paw groom; Group 14 (top right): Locomote (-); 
![Mouse Action Groups 13-16](../examples/group13_16.gif)
#### Group 15 (bottom left): Orient left (+); Group 16 (bottom right): Orient left (-).

More example videos are in [this](../examples) directory .

"""

if __name__ == '__main__':
    print(get_config_str())
    print(f'max_rows_to_read_in_from_csv = {max_rows_to_read_in_from_csv}')
    print(f'VIDEO_FPS = {VIDEO_FPS}')
    print(f'runtime_timestr = {runtime_timestr}')

    pass
