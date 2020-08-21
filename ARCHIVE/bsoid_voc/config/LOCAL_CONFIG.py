################### THINGS YOU MAY WANT TO CHANGE ###################

BASE_PATH = '/Users/ahsu/B-SOID/datasets'  # Base directory path.
TRAIN_FOLDERS = ['/Train1', '/Train2']  # Data folders used to training neural network.
PREDICT_FOLDERS = ['/Data1']  # Data folders, can contain the same as training or new data for consistency.

# Order the points that are encircling the mouth.
BODYPARTS = {
    'Point1': 0,
    'Point2': 1,
    'Point3': 2,
    'Point4': 3,
    'Point5': 4,
    'Point6': 5,
    'Point7': 6,
    'Point8': 7,
}

FPS = 60  # Frame-rate of your video,
# note that you can use a different number for new data as long as the video is same scale/view
COMP = 1  # COMP = 1: Train one classifier for all CSV files; COMP = 0: Classifier/CSV file.

# Output directory to where you want the analysis to be stored
OUTPUT_PATH = '/Users/ahsu/Desktop/bsoid_voc_beta'
# Machine learning model name
MODEL_NAME = 'mky_30min'

# Pick a video
VID_NAME = '/Users/ahsu/B-SOID/datasets/Data1/2019-04-19_09-34-36cut0_30min.mp4'
# What number would be video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)
ID = 0
# Create a folder to store extracted images, make sure this folder exist.
# This program will predict labels and print them on these images
FRAME_DIR = '/Users/ahsu/B-SOID/datasets/Data1/0_30min_10fpsPNGs'
# In addition, this will also create an entire sample group videos for ease of understanding
SHORTVID_DIR = '/Users/ahsu/B-SOID/datasets/Data1/examples'

# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_TRAINING = True
GEN_VIDEOS = True
