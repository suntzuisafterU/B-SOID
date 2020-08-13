"""
TODO: low: purpose
"""

import os


### BSOID PY

# Specify where the OST project lives. Modify on your local machine as necessary.
OST_BASE_PROJECT_PATH = '/home/aaron/Documents/OST-with-DLC'

BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'
# BASE_PATH = os.path.join('C:', 'Users', 'killian', 'projects', 'OST-with-DLC', 'GUI_projects', 'OST-DLC-projects',
#                          'pwd-may11-2020-john-howland-2020-05-11')

# Output directory to where you want the analysis to be stored
OUTPUT_PATH = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT'

MODEL_NAME = 'c57bl6_n3_60min'  # Machine learning model name
# Fetch the base B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Frame-rate of your video,note that you can use a different number for new data as long as the video is same scale/view
FPS = 60

# COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.
COMP = 1

# What number would be video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)
ID = 0

# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_TRAINING = True
GEN_VIDEOS = True


##############################################################################################################
# # # BSDOI UMAP # # #

# IF YOU'D LIKE TO SKIP PLOTS/VIDEOS, change below PLOT/VID settings to False
PLOT = True
VID = True  # if this is true, make sure direct to the video below AND that you created the two specified folders!

# for semi-supervised portion
# CSV_PATH =

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
