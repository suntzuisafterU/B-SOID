### bsoid app
"""
Extracting frames from videos
"""

from typing import List, Tuple
from tqdm import tqdm
import cv2
import glob
import inspect
import numpy as np
import os
import random
import warnings

from bsoid import config
from bsoid.util.likelihoodprocessing import sort_list_nicely_in_place

logger = config.initialize_logger(__name__)


def repeating_numbers(labels) -> Tuple[List, List, List]:  # TODO: low: rename function for clarity
    """
    TODO: med: purpose // purpose unclear
    :param labels: (list) predicted labels
    :return
        n_list: (list) the label number
        idx: (list) label start index
        lengths: (list) how long each bout lasted for
    """
    n_list, idx, lengths = [], [], []
    i = 0
    while i < len(labels) - 1:  # TODO: low: replace with a FOR-loop for clarity?
        current_label = labels[i]
        n_list.append(current_label)
        start_index = i
        idx.append(i)
        while i < len(labels) - 1 and labels[i] == labels[i + 1]:
            i += 1
        end_index = i
        length = end_index - start_index
        lengths.append(length)
        i += 1
    return n_list, idx, lengths


def get_mp4_videos_from_folder_in_BASEPATH(folder_name: str) -> List[str]:
    """
    Previously named `get_video_names()`
    Gets a list of .mp4 files within a folder
    :param folder_name: str, folder path. Must reside in BASE_PATH.
    :return: (List[str]) video file names all of which have a .mp4 extension
    """
    if not isinstance(folder_name, str):
        err = f'`folder_name` was expected to be of type str but instead found {type(folder_name)}.'
        logger.error(err)
        raise TypeError(err)

    path_to_folder = os.path.join(config.DLC_PROJECT_PATH, folder_name)
    path_to_folder_with_glob = f'{path_to_folder}/*.mp4'
    logger.debug(f'get_mp4_videos_from_folder_in_BASEPATH():Path to check for videos: {path_to_folder_with_glob}')
    video_names = glob.glob(path_to_folder_with_glob)
    sort_list_nicely_in_place(video_names)

    return video_names


def write_annotated_frames_to_disk_from_video(path_to_video: str, labels, fps: int, output_path: str = config.FRAME_DIR):
    """
    This function serves to supersede the old 'vid2frame()' function for future clarity.

    Extracts frames every 100ms to match the labels for visualizations  # TODO: Q: are we sure it pulls frames every 100ms when the FPS is variable?
    :param path_to_video: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    # TODO: med: analyze use of magic variables in func.
    """
    if not os.path.isfile(path_to_video):
        err = f'{__name__}:{inspect.stack()[0][3]}Path to video was not found. Path = {path_to_video}'
        logger.error(err)
        raise ValueError(err)
    cv2_video_object = cv2.VideoCapture(path_to_video)
    progress_bar = tqdm(total=int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT)))
    # width = cv2_video_object.get(3)  # TODO: low: address unused variable
    # height = cv2_video_object.get(4)  # TODO: low: address unused variable
    labels = np.hstack((labels[0], labels))  # fill the first frame
    count = 0  # TODO: med: rename `count` -- what is it counting? `i` already tracks iterations over the while loop
    i = 0
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    while cv2_video_object.isOpened():
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            # Prepare writing info to image
            text_to_be_inserted = f'Group{labels[i]}'
            text_width, text_height = cv2.getTextSize(text_to_be_inserted, font, fontScale=font_scale, thickness=1)[0]
            text_offset_x, text_offset_y = 50, 50
            box_coordinates = ((text_offset_x - 12, text_offset_y + 12),
                               (text_offset_x + text_width + 12, text_offset_y - text_height - 8))
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img=frame, text=text_to_be_inserted, org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
            # Write to image
            cv2.imwrite(os.path.join(output_path, f'frame{i}.png'), frame)

            # Save & set metrics, prepare for next frame, and update progress bar
            count += round(fps / 10)  # i.e. at 60fps, this skips every 5
            cv2_video_object.set(1, count)
            progress_bar.update(round(fps / 10))
            i += 1
        else:  # No more frames left to retrieve. Release object and finish.
            cv2_video_object.release()
            break
    progress_bar.close()
    return


def import_vidfolders(folders: List[str], output_path: List[str]):
    """
    Previously called `import_vidfolders()`
    Import multiple folders containing .mp4 files and extract frames from them
    :param folders: list of folder paths
    :param output_path: list, directory to where you want to store extracted vid images in LOCAL_CONFIG
    """
    list_of_lists_of_videos: List[List[str]] = []  # TODO: Q: why does this variable exist? It tracks but does not contribute to anything
    # Loop through folders
    for idx_folder, folder in enumerate(folders):
        videos_list_from_current_folder: List[str] = get_mp4_videos_from_folder_in_BASEPATH(folder)
        # Loop through videos
        for idx_video, video in enumerate(videos_list_from_current_folder):
            logger.info(f'Extracting frames from {video} and appending labels to these images...')
            #
            write_annotated_frames_to_disk_from_video(video, output_path)  # TODO: HIGH: missing param `FPS` *** runtime error imminent ********************************************************
            logger.info(f'Done extracting images and writing labels, from MP4 file {idx_video+1}')
        # After looping through videos, append list of videos from current folder to list of lists because reasons
        list_of_lists_of_videos.append(videos_list_from_current_folder)  # list_of_lists_of_videos.append(videos_list_from_current_folder)
        logger.info(f'Processed {len(videos_list_from_current_folder)} mp4 files from folder: {folder}.')
    return


########################################################################################################################

def create_labeled_vid(labels, crit=3, num_randomly_generated_examples=5,
                       frame_dir=config.FRAME_DIR, output_path=config.SHORTVID_DIR) -> None:
    """
    (Generalized create_labeled_video() function that works between _py, _umap, and _voc submodules)
    TODO: low: purpose
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param num_randomly_generated_examples: scalar, number of randomly generated examples, default 5  # TODO: low: default is actually 3..does counts refer to cv2.VideoWriter pathing?
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    # Create list of only .png images found in `frame_dir`
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_list_nicely_in_place(images)
    four_character_code = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    ranges_list, idx2_list = [], []
    n, idx, lengths = repeating_numbers(labels)
    #
    for idx_length, length in enumerate(lengths):
        if length >= crit:
            ranges_list.append(range(idx[idx_length], idx[idx_length] + length))
            idx2_list.append(idx_length)

    # Loop over the range generated from the total unique labels available
    for idx_label in tqdm(range(len(np.unique(labels)))):
        a = []  # TODO: low: `a` needs more description
        for idx_range in range(len(ranges_list)):
            if n[idx2_list[idx_range]] == idx_label:
                a += [ranges_list[idx_range], ]  # Previously: a.append(ranges[idx_range]). Remove comment as necessary.
        try:
            random_ranges = random.sample(a, num_randomly_generated_examples)  # TODO: add a min() function to `counts` argument?
            for idx_random_range in range(len(random_ranges)):
                grp_images = []
                video_name = f'group_{idx_label}_example_{idx_random_range}.mp4'
                # Loop over list of randomly generated ranges
                for a_random_range in random_ranges[idx_random_range]:
                    # Aggregate images into a list that correspond to the randomly generated numbers/ranges
                    grp_images += [images[a_random_range], ]  # grp_images.append(images[random_range])
                grp_images = []
                # Open video writer
                video_writer = cv2.VideoWriter(
                    os.path.join(output_path, video_name),
                    four_character_code,
                    5,  # TODO: med: 5 is a magic variable? FPS?
                    (width, height))
                # Loop over all images and write to file
                for image in grp_images:
                    video_writer.write(cv2.imread(os.path.join(frame_dir, image)))
                # Release and continue
                cv2.destroyAllWindows()
                video_writer.release()
        except:  # TODO: low: exception is very general. Address?
            pass
    return


def create_labeled_vid_app(labels, crit, counts, output_fps, video_frames_directory, output_path) -> None:
    """
    *** LEGACY WARNING: this function is different from the non-annotated (_py/_umap/_voc)
    implementation(s) ONLY with regards to the new parameter `output_fps`.
    Since the two functions have not yet been reconciled, this function remains as
    legacy. It is not used in the submodule bsoid_app codebase but is used 2 times in .ipynb files ***

    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param video_frames_directory: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(video_frames_directory) if img.endswith(".png")]
    sort_list_nicely_in_place(images)
    four_character_code = cv2.VideoWriter_fourcc(*'avc1')  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(video_frames_directory, images[0]))
    height, width, layers = frame.shape
    ranges, idx2 = [], []
    n, idx, lengths = repeating_numbers(labels)
    for idx_length, length in enumerate(lengths):
        if length >= crit:
            ranges.append(range(idx[idx_length], idx[idx_length] + length))
            idx2.append(idx_length)
    for label in tqdm(np.unique(labels)):
        # a = []
        # for j in range(len(ranges)):
        #     if n[idx2[j]] == label:
        #         a.append(ranges[j])
        a = [ranges[i] for i in range(len(ranges)) if n[idx2[i]] == label]
        try:
            random_ranges = random.sample(a, min(len(a), counts))
            for k in range(len(random_ranges)):
                # grp_images = []
                # for random_range in random_ranges[k]:
                #     grp_images.append(images[random_range])
                grp_images = [images[random_range] for random_range in random_ranges[k]]
                video_name = f'group_{label}_example_{k}.mp4'
                video = cv2.VideoWriter(os.path.join(output_path, video_name),
                                        four_character_code, output_fps, (width, height))
                for image in grp_images:
                    video.write(cv2.imread(os.path.join(video_frames_directory, image)))
                cv2.destroyAllWindows()
                video.release()
        except:  # TODO: low: exception is very general. Address?
            pass
    return

########################################################################################################################


def get_frames_from_video_then_create_labeled_video(path_to_video, labels, fps, output_path) -> None:
    """ # TODO: rename function for concision/clarity
    TODO: Purpose
    :param path_to_video: (str) TODO
    :param labels: TODO
    :param fps: (int) TODO
    :param output_path: TODO
    :return:
    """
    write_annotated_frames_to_disk_from_video(path_to_video, labels, fps, output_path)
    create_labeled_vid(labels, crit=3, num_randomly_generated_examples=5, frame_dir=output_path, output_path=config.SHORTVID_DIR)


###
# Legacy functions (some of which will be deprecated)
def vid2frame(path_to_video: str, labels, fps: int, output_path: str = config.FRAME_DIR):
    """ # # # DEPRECATION WARNING # # # """
    replacement_func = write_annotated_frames_to_disk_from_video
    warnings.warn(f'This function, vid2frame(), will be deprecated shortly. The replacement '
                  f'function is called "{replacement_func.__qualname__}" and aims to make usage more clear and DRY. '
                  f'If you are reading this, this function was kept for backwards compatibility reasons. ')
    return replacement_func(path_to_video, labels, fps, output_path)


def main(path_to_video, labels, fps, output_path):  # To be deprecated
    """# # # DEPRECATION WARNING # # #"""
    replacement_function = get_frames_from_video_then_create_labeled_video
    warnings.warn('This function, bsoid.util.videoprocessing.main(), will be deprecated in the future in '
                  'favour of a refactored, more descriptive function. Currently, that function is: '
                  f'{replacement_function.__qualname__}')
    return replacement_function(path_to_video, labels, fps, output_path)
