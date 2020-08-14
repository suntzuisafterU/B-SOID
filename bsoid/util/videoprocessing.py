### bsoid app
"""
Extracting frames from videos
"""

from typing import List, Tuple
from tqdm import tqdm
import cv2
import glob
import logging
import numpy as np
import os
import random

from bsoid.config.LOCAL_CONFIG import BASE_PATH, FRAME_DIR, SHORTVID_DIR
from bsoid.util.likelihoodprocessing import sort_nicely


def repeatingNumbers(labels) -> Tuple:  # TODO: rename function for clarity
    """
    TODO: med: purpose
    :param labels: 1D array, predicted labels
    :return n_list: 1D array, the label number
    :return idx: 1D array, label start index
    :return lengths: 1D array, how long each bout lasted for
    """
    i = 0
    n_list, idx, lengths = [], [], []
    while i < len(labels) - 1:
        n = labels[i]
        n_list.append(n)
        start_index = i
        idx.append(i)
        while i < len(labels) - 1 and labels[i] == labels[i + 1]:
            i = i + 1
        end_index = i
        length = end_index - start_index
        lengths.append(length)
        i += 1
    return n_list, idx, lengths


def get_video_names(folder_name) -> List[str]:
    """
    Gets a list of .mp4 files within a folder
    :param folder_name: str, folder path. Must reside in BASE_PATH.
    :return: list, video filenames
    """
    # TODO: low: stretch goal: ensure this function works independent of OS
    path_to_folder = os.path.join(BASE_PATH, folder_name)
    # video_names = glob.glob(BASE_PATH + folder_name + '/*.mp4')
    video_names = glob.glob(f'{path_to_folder}/*.mp4')
    sort_nicely(video_names)
    return video_names


def vid2frame(path_to_video, labels, fps, output_path=FRAME_DIR):
    """
    Extracts frames every 100ms to match the labels for visualizations
    :param path_to_video: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    """
    cv2_video_object = cv2.VideoCapture(path_to_video)
    progress_bar = tqdm(total=int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = cv2_video_object.get(3)  # TODO: low: unused variable
    height = cv2_video_object.get(4)  # TODO: low: unused variable
    labels = np.hstack((labels[0], labels))  # fill the first frame
    count = 0  # TODO: med: rename `count` -- what is it counting? i already tracks iterations over the while loop
    i = 0
    font_scale = 1
    font = cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    while cv2_video_object.isOpened():
        ret, frame = cv2_video_object.read()
        if ret:
            text = 'Group' + str(labels[i])
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
            text_offset_x, text_offset_y = 50, 50  # TODO: med: magic variables reside here
            box_coordinates = ((text_offset_x - 12, text_offset_y + 12),
                               (text_offset_x + text_width + 12, text_offset_y - text_height - 8))
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                        fontScale=font_scale, color=(255, 255, 255), thickness=1)
            cv2.imwrite(os.path.join(output_path, 'frame{:d}.png'.format(i)), frame)
            count += round(fps / 10)  # i.e. at 60fps, this skips every 5
            i += 1
            cv2_video_object.set(1, count)
            progress_bar.update(round(fps / 10))
        else:
            cv2_video_object.release()
            break
    progress_bar.close()
    return


def import_vidfolders(folders: List[str], output_path: List[str]):
    """
    Import multiple folders containing .mp4 files and extract frames from them
    :param folders: list of folder paths
    :param output_path: list, directory to where you want to store extracted vid images in LOCAL_CONFIG
    """
    video_names = []
    for idx, folder in enumerate(folders):  # Loop through folders
        v = get_video_names(folder)
        for j, video_name in enumerate(v):
            logging.info(f'Extracting frames from {video_name} and appending labels to these images...')
            vid2frame(video_name, output_path)  # TODO: HIGH: missing param `FPS`
            logging.info(f'Done extracting images and writing labels, from MP4 file {j+1}')
        video_names.append(v)
        logging.info(f'Processed {len(v)} MP4 files from folder: {folder}')
    return


# TODO: HIGH: disambiguate between create_labeled_video
def create_labeled_vid(labels, crit=3, counts=3, frame_dir=FRAME_DIR, output_path=SHORTVID_DIR):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    rnges = []
    n, idx, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            rnges.append(range(idx[i], idx[i] + j))
            idx2.append(i)
    for i in tqdm(range(0, len(np.unique(labels)))):
        a = []
        for j in range(0, len(rnges)):
            if n[idx2[j]] == i:
                a.append(rnges[j])
        try:
            rand_rnges = random.sample(a, counts)
            for k in range(0, len(rand_rnges)):
                video_name = 'group_{}_example_{}.mp4'.format(i, k)
                grpimages = []
                for l in rand_rnges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 5, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except:  # TODO: low: exception is very general. Address?
            pass
    return
def create_labeled_vid(labels, crit=3, counts=5, frame_dir=FRAME_DIR, output_path=SHORTVID_DIR):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    rnges = []
    n, idx, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            rnges.append(range(idx[i], idx[i] + j))
            idx2.append(i)
    for b, i in enumerate(tqdm(np.unique(labels))):
        a = []
        for j in range(0, len(rnges)):
            if n[idx2[j]] == i:
                a.append(rnges[j])
        try:
            rand_rnges = random.sample(a, counts)
            for k in range(0, len(rand_rnges)):
                video_name = 'group_{}_example_{}.mp4'.format(i, k)
                grpimages = []
                for l in rand_rnges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 5, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except:
            pass
    return


# TODO: high disambiguate between create_labeled_vid() functions
def create_labeled_vid(labels, crit, counts, output_fps, frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    ranges = []
    n, idx, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            ranges.append(range(idx[i], idx[i] + j))
            idx2.append(i)
    for b, i in enumerate(tqdm(np.unique(labels))):
        a = []
        for j in range(len(ranges)):
            if n[idx2[j]] == i:
                a.append(ranges[j])
        try:
            random_ranges = random.sample(a, min(len(a), counts))
            for k in range(len(random_ranges)):
                video_name = f'group_{i}_example_{k}.mp4'
                grpimages = []
                for l in random_ranges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except: pass  # TODO: med: exception too broad
    return
def create_labeled_vid(labels, crit=3, counts=3, frame_dir=FRAME_DIR, output_path=SHORTVID_DIR):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    ranges = []
    n, index_list, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            ranges.append(range(index_list[i], index_list[i]+j))
            idx2.append(i)
    for i in tqdm(range(len(np.unique(labels)))):
        
        a = []
        for j in range(len(ranges)):
            if n[idx2[j]] == i:
                a.append(ranges[j])
        try:
            random_ranges = random.sample(a, counts)
            for k in range(len(random_ranges)):
                video_name = f'group_{i}_example_{k}.mp4'
                grpimages = []
                for l in random_ranges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 5, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except:  # TODO: med: exception too broad
            pass
    return


def main(path_to_video, labels, fps, output_path):
    vid2frame(path_to_video, labels, fps, output_path)
    create_labeled_vid(labels, crit=3, counts=5, frame_dir=output_path, output_path=SHORTVID_DIR)


