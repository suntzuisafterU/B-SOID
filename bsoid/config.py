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

from ast import literal_eval  # TODO: HIGH HIGH HIGH: use this to read in tuples from config file
import configparser
import logging
import os
import random
import sys

from bsoid.config import LOCAL_CONFIG, GLOBAL_CONFIG
from bsoid.util import logger_config

randomly_generated_state_int: int = random.randint(1, 100_000)

debug = 1

###
# Load up a configuration logger to detect config problems. Otherwise, use the general logger


# config_logger = logging.Logger()


########################################################################################################################

# Fetch the B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if debug:
    print(BSOID_BASE_PROJECT_PATH)

# Load up config file
configuration = configparser.ConfigParser()
configuration.read(os.path.join(BSOID_BASE_PROJECT_PATH, 'config.ini'))


# Specify where the OST project lives. Modify on your local machine as necessary.
OST_BASE_PROJECT_PATH = configuration['PATH']['OST_BASE_PROJECT_PATH']  # 'previously: /home/aaron/Documents/OST-with-DLC'
# OST_BASE_PROJECT_PATH = os.path.join('C:', 'Users', 'killian', 'projects', 'OST-with-DLC')


if debug: print(configuration['LOGGING']['DEFAULT_LOG_FILE'])
if debug: print('OST PATH:', configuration.get('PATH', 'OSTPATH', fallback=None))

# Instantiate logger
bsoid_logger = logger_config.create_generic_logger(
    logger_name=configuration['LOGGING']['LOG_FILE_NAME'],
    log_format=configuration['LOGGING']['LOG_FORMAT'],
    stdout_log_level=configuration.get('LOGGING', 'STREAM_LOG_LEVEL', fallback=None),

)


holdout_percent: float = configuration.getfloat('APP', 'HOLDOUT_TEST_PCT')
kfold_crossvalidation: int = configuration.getint('APP', 'CROSS_VALIDATION_K')  # Number of iterations for cross-validation to show it's not over-fitting.
fps_video: int = configuration.getint('APP', 'FRAME_RATE_VIDEO')  # ['APP']['FRAME_RATE_VIDEO']
COMPILE_CSVS_FOR_TRAINING = int(configuration['APP']['COMPILE_CSVS_FOR_TRAINING'])

########################################################################################################################
# LEGACY VARIABLES

HLDOUT: float = holdout_percent  # Test partition ratio to validate clustering separation.
CV_IT: int = kfold_crossvalidation


# Frame-rate of your video,note that you can use a different number for new data as long as the video is same scale/view
FPS = int(configuration['APP']['FRAME_RATE_VIDEO'])  # TODO: med: deprecate

# COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.
COMP: int = COMPILE_CSVS_FOR_TRAINING  # TODO: med: deprecate

# What number would be video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)
ID = 0

# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_TRAINING = True
GEN_VIDEOS = True


########################################################################################################################


# BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'
BASE_PATH = os.path.join('C:\\', 'Users', 'killian', 'projects', 'OST-with-DLC', 'pwd-may11-2020-john-howland-2020-05-11')

# Output directory to where you want the analysis to be stored
# OUTPUT_PATH = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT'
OUTPUT_PATH = os.path.join('C:\\', 'Users', 'killian', 'Pictures')

MODEL_NAME = 'c57bl6_n3_60min'  # Machine learning model name


# TODO: med: for TRAIN_FOLDERS & PREDICT_FOLDERS, change path resolution from inside functional module to inside this config file
# Data folders used to training neural network.
TRAIN_FOLDERS = [os.path.sep+'training_datasets', ]

# Data folders, can contain the same as training or new data for consistency.
PREDICT_FOLDERS = [os.path.sep+'Data1', ]

# Create a folder to store extracted images, MAKE SURE THIS FOLDER EXISTS.  # TODO: med: add in a runtime check that folder exists
FRAME_DIR = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT', 'frames')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'

# Create a folder to store created video snippets/group, MAKE SURE THIS FOLDER EXISTS.  # TODO: med: add in a runtime check that folder exists
# Create a folder to store extracted images, make sure this folder exist.
#   This program will predict labels and print them on these images
# In addition, this will also create an entire sample group videos for ease of understanding
SHORTVID_DIR = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT', 'shortvids')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/shortvids'

# Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS
VID_NAME = os.path.join(OST_BASE_PROJECT_PATH, 'GUI_projects', 'labelled_videos', '002_ratA_inc2_above.mp4')  # '/home/aaron/Documents/OST-with-DLC/GUI_projects/labelled_videos/002_ratA_inc2_above.mp4'


# This version requires the six body parts Snout/Head, Forepaws/Shoulders, Hindpaws/Hips, Tailbase.
BODYPARTS = {
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


##############################################################################################################
### BSOID VOC ###
# TODO: HIGHaddress BODYPARTS variable also found in bsoid_voc. Does it do the same as _py? Naming collision.
# # Order the points that are encircling the mouth.
# BODYPARTS = {
#     'Point1': 0,
#     'Point2': 1,
#     'Point3': 2,
#     'Point4': 3,
#     'Point5': 4,
#     'Point6': 5,
#     'Point7': 6,
#     'Point8': 7,
# }


#################
#BSOIDAPP

# BSOID UMAP params, nonlinear transform
UMAP_PARAMS = {
    'n_components': configuration.getint('UMAP', 'n_components'),
    'min_dist': configuration.getfloat('UMAP', 'min_dist'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=randomly_generated_state_int),
}

# HDBSCAN params, density based clustering
HDBSCAN_PARAMS = {
    'min_samples': configuration.getint('HDBSCAN', 'min_samples')  # small value
}

# Feedforward neural network (MLP) params
MLP_PARAMS = {
    'hidden_layer_sizes': literal_eval(configuration.get('MLP', 'hidden_layer_sizes')),
    'activation': configuration.get('MLP', 'activation'),
    'solver': configuration.get('MLP', 'solver'),
    'learning_rate': configuration.getfloat('MLP', 'learning_rate'),
    'learning_rate_init': configuration.getfloat('MLP', 'learning_rate_init'),
    'alpha': configuration.getfloat('MLP', 'alpha'),
    'max_iter': configuration.getint('MLP', 'max_iter'),
    'early_stopping': configuration.getboolean('MLP', 'early_stopping'),
    'verbose': configuration.getint('MLP', 'verbose'),
}


########################################################################################################################
####### BSOIDPY

# EM_GMM parameters
EMGMM_PARAMS = {
    'n_components': configuration.getint('EM/GMM', 'n_components'),
    'covariance_type': configuration.get('EM/GMM', 'covariance_type'),
    'tol': configuration.getfloat('EM/GMM', 'tol'),
    'reg_covar': configuration.getfloat('EM/GMM', 'reg_covar'),
    'max_iter': configuration.getint('EM/GMM', 'max_iter'),
    'n_init': configuration.getint('EM/GMM', 'n_init'),
    'init_params': configuration.get('EMM/GMM', 'init_params'),
    'verbose': configuration.getint('EM/GMM', 'verbose'),
    'random_state': configuration.getint('MODEL', 'RANDOM_STATE', fallback=randomly_generated_state_int),
}
# Multi-class support vector machine classifier params
SVM_PARAMS = {
    'C': configuration.getint('SVM', 'C'),
    'gamma': configuration.getfloat('SVM', 'gamma'),
    'probability': configuration.getboolean('SVM', 'probability'),
    'verbose': configuration.getint('SVM', 'verbose'),
    'random_state': configuration.getint('APP', 'RANDOM_STATE', fallback=randomly_generated_state_int),
}


########################################################################################################################
# # # BSDOI UMAP # # #

# IF YOU'D LIKE TO SKIP PLOTS/VIDEOS, change below PLOT/VID settings to False
PLOT_GRAPHS: bool = True  # New variable name for `PLOT`
PLOT = PLOT_GRAPHS  # `PLOT` is likely to be deprecated in the future
PRODUCE_VIDEO: bool = True
VID = PRODUCE_VIDEO  # if this is true, make sure direct to the video below AND that you created the two specified folders!

# for semi-supervised portion
# CSV_PATH =

########################################################################################################################
### BSOID VOC
# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)
TSNE_PARAMS = {
    'n_components': configuration.getint('TSNE', 'n_components'),
    'random_state': configuration.getint('APP', 'RANDOM_STATE'),
    'n_jobs': configuration.getint('TSNE', 'n_jobs'),
    'verbose': configuration.getint('TSNE', 'verbose'),
}




# TODO: HIGH: after all config parsing done, write checks that will run ON IMPORT to ensure folders exist :)


