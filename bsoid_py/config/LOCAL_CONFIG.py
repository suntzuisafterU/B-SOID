################### THINGS YOU MAY WANT TO CHANGE ###################
import os

# Specify where the OST project lives. Modify on your local machine as necessary.
OST_BASE_PROJECT_PATH = '/home/aaron/Documents/OST-with-DLC'

BASE_PATH = '/home/aaron/Documents/OST-with-DLC/GUI_projects/OST-DLC-projects/pwd-may11-2020-john-howland-2020-05-11'
# BASE_PATH = os.path.join('C:', 'Users', 'killian', 'projects', 'OST-with-DLC', 'GUI_projects', 'OST-DLC-projects',
#                          'pwd-may11-2020-john-howland-2020-05-11')

# Fetch the base B-SOiD project directory regardless of clone location
BSOID_BASE_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: for the 2 below variables, change path resolution from inside other module to inside this config file
TRAIN_FOLDERS = [os.path.sep+'training_datasets']  # Data folders used to training neural network.
PREDICT_FOLDERS = [os.path.sep+'Data1']  # Data folders, can contain the same as training or new data for consistency.

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

FPS = 60  # Frame-rate of your video,
# note that you can use a different number for new data as long as the video is same scale/view

COMP = 1  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.

# Output directory to where you want the analysis to be stored
OUTPUT_PATH = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT'
# Machine learning model name
MODEL_NAME = 'my_first_model'

# Pick a video
VID_NAME = os.path.join(OST_BASE_PROJECT_PATH, 'GUI_projects', 'labelled_videos', '002_ratA_inc2_above.mp4')  # '/home/aaron/Documents/OST-with-DLC/GUI_projects/labelled_videos/002_ratA_inc2_above.mp4'

# What number would be video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)
ID = 0

# Create a folder to store extracted images, make sure this folder exist.
# This program will predict labels and print them on these images
FRAME_DIR = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT', 'frames')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/frames'
# In addition, this will also create an entire sample group videos for ease of understanding
SHORTVID_DIR = os.path.join(OST_BASE_PROJECT_PATH, 'B-SOID', 'OUTPUT', 'shortvids')  # '/home/aaron/Documents/OST-with-DLC/B-SOID/OUTPUT/shortvids'

# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_TRAINING = True
GEN_VIDEOS = True
