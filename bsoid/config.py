"""
TODO: low: purpose

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
import sys


from bsoid.util import logger_config


# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level='INFO', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)


debug = 0  # TODO: delete me after debugging and implementation is done.


# Fetch the B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# logging.info(f'BSOID_BASE_PROJECT_PATH:::{BSOID_BASE_PROJECT_PATH}')
# Output directory to where you want the analysis to be stored
OUTPUT_PATH = os.path.join(BSOID_BASE_PROJECT_PATH, 'output')
# Set loggers default vars
default_log_folder_path = Path(BSOID_BASE_PROJECT_PATH, 'logs').absolute()
# logging.info(f'default_log_folder_path:::{default_log_folder_path}')
default_log_file_name = 'default.log'
# set default config file name
config_file_name = 'config.ini'
# Load up config file
configuration = configparser.ConfigParser()
configuration.read(os.path.join(BSOID_BASE_PROJECT_PATH, config_file_name))

# Load up a configuration logger to detect config problems. Otherwise, use the general logger  # TODO
# config_logger = logging.Logger()

# Instantiate runtime variables
random_state: int = configuration.getint('MODEL', 'RANDOM_STATE', fallback=random.randint(1, 100_000_000))
holdout_percent: float = configuration.getfloat('MODEL', 'HOLDOUT_TEST_PCT')
crossvalidation_k: int = configuration.getint('MODEL', 'CROSS_VALIDATION_K')  # Number of iterations for cross-validation to show it's not over-fitting.
crossvalidation_n_jobs: int = configuration.getint('MODEL', 'CROSS_VALIDATION_N_JOBS')
video_fps: int = configuration.getint('APP', 'VIDEO_FRAME_RATE')  # ['APP']['VIDEO_FRAME_RATE']
compile_CSVs_for_training: int = configuration.getint('APP', 'COMPILE_CSVS_FOR_TRAINING')  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.
identification_order: int = configuration.getint('APP', 'FILE_IDENTIFICATION_ORDER_LEGACY')  # TODO: low: assess whether we can remove this from module altogether.
# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_GRAPHS: bool = configuration.getboolean('APP', 'PLOT_GRAPHS')
PLOT_TRAINING: bool = configuration.getboolean('APP', 'PLOT_TRAINING')
GENERATE_VIDEOS: bool = configuration.getboolean('APP', 'GENERATE_VIDEOS')
PRODUCE_VIDEO: bool = configuration.getboolean('APP', 'PRODUCE_VIDEO')



# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'
# BASE_PATH = os.path.join('C:\\', 'Users', 'killian', 'projects', 'OST-with-DLC', 'pwd-may11-2020-john-howland-2020-05-11')
BASE_PATH = 'C:\\Users\\killian\\projects\\OST-with-DLC\\GUI_projects\\OST-DLC-projects\\pwd-may11-2020-john-howland-2020-05-11'
# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'

MODEL_NAME = configuration.get('APP', 'OUTPUT_MODEL_NAME')  # Machine learning model name

# TODO: med: for TRAIN_FOLDERS & PREDICT_FOLDERS, change path resolution from inside functional module to inside this config file
# Data folders used to training neural network.
# TRAIN_FOLDERS = [os.path.sep+'training-datasets', ]
TRAIN_FOLDERS: List[str] = ['NOT_DLC_OUTPUT__SAMPLE_WITH_INDEX', ]

# Data folders, can contain the same as training or new data for consistency.
PREDICT_FOLDERS = [os.path.sep+'Data1', ]

# Create a folder to store extracted images, MAKE SURE THIS FOLDER EXISTS.  # TODO: med: add in a runtime check that folder exists
FRAME_DIR = os.path.join(OUTPUT_PATH, 'frames')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'

# Create a folder to store created video snippets/group, MAKE SURE THIS FOLDER EXISTS.  # TODO: med: add in a runtime check that folder exists
# Create a folder to store extracted images, make sure this folder exist.
#   This program will predict labels and print them on these images
# In addition, this will also create an entire sample group videos for ease of understanding
SHORTVID_DIR = os.path.join(OUTPUT_PATH, 'shortvids')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/shortvids'
# assert os.path.isdir(SHORTVID_DIR), f'`SHORTVID` dir. (value={SHORTVID_DIR}) must exist for runtime but does not.'

# Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS

# VID_NAME = os.path.join(OST_BASE_PROJECT_PATH, 'GUI_projects', 'labelled_videos', '002_ratA_inc2_above.mp4')  # '/home/aaron/Documents/OST-with-DLC/GUI_projects/labelled_videos/002_ratA_inc2_above.mp4'







# GEN_VIDEOS = GENERATE_VIDEOS  # Deprecate GEN_VIDEOS
# FPS = video_fps  # DEPRECATE # FPS: Frame-rate of your video,note that you can use a different number for new data as long as the video is same scale/view
# HLDOUT: float = holdout_percent  # Test partition ratio to validate clustering separation.  # DEPRECATE
# CV_IT: int = kfold_crossvalidation  # DEPRECATE
# COMP: int = compile_CSVs_for_training  # TODO: med: deprecate
# ID = identification_order  # DEPRECATE


########################################################################################################################

# Specify where the OST project lives. Modify on your local machine as necessary.
OST_BASE_PROJECT_PATH = configuration.get('PATH', 'OST_BASE_PROJECT_PATH')  # 'previously: /home/aaron/Documents/OST-with-DLC'
# OST_BASE_PROJECT_PATH = os.path.join('C:\\', 'Users', 'killian', 'projects', 'OST-with-DLC')
# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'


########################################################################################################################

# logging.info(f"configuration.get('LOGGING', 'LOG_FILE_NAME'):::{configuration.get('LOGGING', 'LOG_FILE_NAME')}")

# if debug == 2: print('OST PATH:::', configuration.get('PATH', 'OSTPATH', fallback=None))

# Resolve logger variables
config_file_log_folder_path = configuration.get('LOGGING', 'LOG_FILE_FOLDER_PATH')
config_file_log_folder_path = config_file_log_folder_path if config_file_log_folder_path else default_log_folder_path
if debug >= 2: print('config_file_log_folder_path:::', config_file_log_folder_path)

config_file_name = configuration.get('LOGGING', 'LOG_FILE_NAME', fallback=default_log_file_name)
if debug >= 2: print('config_file_name:::', config_file_name)

log_file_file_path = str(Path(config_file_log_folder_path, config_file_name).absolute())
if debug >= 2: print('log_file_file_path AKA os.path.join(config_file_log_folder_path, config_file_name):::', log_file_file_path)

assert os.path.isdir(config_file_log_folder_path), f'Path does not exist: {config_file_log_folder_path}'

# Instantiate logger
bsoid_logger = logger_config.create_generic_logger(
    logger_name=configuration.get('LOGGING', 'LOGGER_NAME'),   # configuration['LOGGING']['LOG_FILE_NAME'],
    log_format=configuration.get('LOGGING', 'LOG_FORMAT', raw=True),
    stdout_log_level=configuration.get('LOGGING', 'STREAM_LOG_LEVEL', fallback=None),
    file_log_level=configuration.get('LOGGING', 'FILE_LOG_LEVEL', fallback=None),
    file_log_file_path=log_file_file_path,
)


##############################################################################################################
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


########################
### MODEL PARAMETERS ###

UMAP_PARAMS = {
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=random_state),
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
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=random_state),
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
    'random_state': configuration.getint('APP', 'RANDOM_STATE', fallback=random_state),
}


########################################################################################################################
### BSOID VOC
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
# LEGACY VARIABLES


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
