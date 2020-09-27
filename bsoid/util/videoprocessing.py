### bsoid app
"""
Extracting frames from videos
"""

from typing import Any, List, Tuple, Union
from tqdm import tqdm
import cv2
import glob
import inspect
import multiprocessing
import numpy as np
import os
import random

from bsoid import config
from bsoid.util import likelihoodprocessing
# from bsoid.util.likelihoodprocessing import likelihoodprocessing.sort_list_nicely_in_place

logger = config.initialize_logger(__name__)


########################################################################################################################

@config.deco__log_entry_exit(logger)
def augmented_runlength_encoding(labels: Union[List, np.ndarray]) -> Tuple[List[Any], List[int], List[int]]:  # TODO: low: rename function for clarity
    """
    TODO: med: purpose // purpose unclear
    :param labels: (list or np.ndarray) predicted labels
    :return
        label_list: (list) the label number
        idx: (list) label start index
        lengths: (list) how long each bout lasted for
    """
    warning = f'The version for RLE has moved from video processing. Caller = {inspect.stack()[0][3]}().' \
              f'Ensure that the caller changes from videoprocessing version to likelihood'
    logger.warning(warning)
    return likelihoodprocessing.augmented_runlength_encoding(labels)
    label_list, idx_list, lengths_list = [], [], []
    i = 0
    while i < len(labels) - 1:
        # 1/3: Record current index
        idx_list.append(i)
        # 2/3: Record current label
        current_label = labels[i]
        label_list.append(current_label)
        # Iterate over i while current label and next label are same
        start_index = i
        while i < len(labels)-1 and labels[i] == labels[i + 1]:
            i += 1
        end_index = i
        # 3/3: Calculate length of repetitions, then record lengths to list
        length = end_index - start_index
        lengths_list.append(length)
        # Increment and continue
        i += 1

    return label_list, idx_list, lengths_list


def get_videos_from_folder_in_BASEPATH(folder_name: str, video_extension: str = 'mp4') -> List[str]:
    """
    Previously named `get_video_names()`
    Gets a list of video files within a folder
    :param folder_name: str, folder path. Must reside in BASE_PATH.
    :param video_extension:
    :return: (List[str]) video file names all of which have a .mp4 extension
    """
    if not isinstance(folder_name, str):
        err = f'`{inspect.stack()[0][3]}(): folder_name was expected to be of ' \
              f'type str but instead found {type(folder_name)}.'
        logger.error(err)
        raise TypeError(err)
    path_to_folder = os.path.join(config.DLC_PROJECT_PATH, folder_name)

    path_to_folder_with_glob = f'{path_to_folder}/*.{video_extension}'
    logger.debug(f'{inspect.stack()[0][3]}(): Path to check for videos: {path_to_folder_with_glob}.')
    video_names = glob.glob(path_to_folder_with_glob)
    likelihoodprocessing.sort_list_nicely_in_place(video_names)
    return video_names


def write_frame_to_file(is_frame_retrieved: bool, frame: object, label, frame_idx, frames_to_skip_after_each_write, output_path=config.FRAMES_OUTPUT_PATH):
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    if is_frame_retrieved:
        # Prepare writing info onto image
        text_for_frame = f'Label__'
        # Try appending label
        try:
            label_word = config.map_group_to_behaviour[label]
            text_for_frame += label_word
        except KeyError:
            text_for_frame += f'NotFound. Group: {label}'
            label_not_found_err = f'Label number not found: {label}. '
            logger.error(label_not_found_err)
        except IndexError as ie:
            index_err = f'Index error. Could not index i ({frame_idx}) onto labels. / ' \
                        f'is_frame_retrieved = {is_frame_retrieved} / ' \
                        f'Original exception: {repr(ie)}'
            logger.error(index_err)
            raise IndexError(index_err)
        else:
            text_width, text_height = cv2.getTextSize(
                text_for_frame, font, fontScale=font_scale, thickness=1)[0]
            # TODO: evaluate magic variables RE: text offsetting on images
            text_offset_x, text_offset_y = 50, 50
            box_coordinates = (
                (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
                (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
            )
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img=frame, text=text_for_frame, org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
            # Write to image
            image_name = f'frame_{frame_idx + 1}.png'
            cv2.imwrite(os.path.join(output_path, image_name), frame)

        # Save & set metrics, prepare for next frame, and update progress bar
        # frame_count += frames_to_skip_after_each_write
        # cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
        # progress_bar.update(frames_to_skip_after_each_write)
        # i += 1
    else:  # No more frames left to retrieve. Release object and finish.
        pass
    # return progress_bar, frames_to_skip_after_each_write


@config.deco__log_entry_exit(logger)
def write_annotated_frames_to_disk_from_video_NEW_multiproc(path_to_video: str, labels, fps: int = config.VIDEO_FPS, output_path: str = config.FRAMES_OUTPUT_PATH, pct_frames_to_label: float = config.PERCENT_FRAMES_TO_LABEL):
    """ *new*
    New implementation to leverage multiprocessing (optional) just because original implementation is so slow.
    :param path_to_video:
    :param labels:
    :param fps:
    :param output_path:
    :param pct_frames_to_label:
    :return:
    """

    assert os.path.isfile(path_to_video), f'Video does not exist: {path_to_video}'

    frames_to_skip_after_each_write = round(1 / pct_frames_to_label)
    set_unique_labels = set(np.unique(labels))

    cv2_video_object = cv2.VideoCapture(path_to_video)
    total_frames = int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f'Total frames: {total_frames}')
    logger.debug(f'Initial labels shape: {labels.shape}')
    # progress_bar = tqdm(total=total_frames, desc='Writing images to file...')
    # width, height = cv2_video_object.get(3), cv2_video_object.get(4)  # TODO: low: address unused variables
    labels = np.hstack((labels[0], labels))  # fill the first frame  # TODO: why need to fill first frame?
    logger.debug(f'Labels shape after padding: {labels.shape}')
    frame_count = 0  # TODO: med: rename `count` -- what is it counting? Other variable `i` already tracks iterations over the while loop
    i = 0
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    frames_queue: List[Tuple[bool, Any, str, int]] = []
    # queue up images to write
    done = False
    # while cv2_video_object.isOpened():
    while not done:
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            frames_queue.append((is_frame_retrieved, frame, labels[i], frame_count))
            # Save & set metrics, prepare for next frame, and update progress bar
            frame_count += frames_to_skip_after_each_write
            cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
            # progress_bar.update(frames_to_skip_after_each_write)
            i += 1
        else:  # No more frames left to retrieve. Release object and finish.
            done = True
            cv2_video_object.release()
            break
    #
    with multiprocessing.Pool(config.N_JOBS) as pool:
        # Set up function to be executed async
        results = [pool.apply_async(
            write_frame_to_file,
            (is_frame_retrieved, frame, label, i, frames_to_skip_after_each_write),
        )
            for is_frame_retrieved, frame, label, i in frames_queue]
        # Execute
        results = [res.get() for res in results]

    cv2_video_object.release()
    # progress_bar.close()
    return


@config.deco__log_entry_exit(logger)
def write_annotated_frames_to_disk_from_video(path_to_video: str, labels, fps: int = config.VIDEO_FPS, output_path: str = config.FRAMES_OUTPUT_PATH, pct_frames_to_label: float = config.PERCENT_FRAMES_TO_LABEL):
    """
    This function serves to supersede the old 'vid2frame()' function for future clarity.

    Extracts frames every 100ms to match the labels for visualizations  # TODO: Q: are we sure it pulls frames every 100ms when the FPS is variable?

    Assumptions:
        -

    :param path_to_video: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    :param pct_frames_to_label:
    # TODO: med: analyze use of magic variables in func.
    """
    if not os.path.isfile(path_to_video):  # Check if path to video exists.
        err = f'{__name__}:{inspect.stack()[0][3]}Path to video was not found. Path = {path_to_video}'
        logger.error(err)
        raise ValueError(err)

    #
    frames_to_skip_after_each_write = round(1 / pct_frames_to_label)  # frames_to_skip_after_each_write = round(fps * (1 / pct_frames_to_label))  # TODO: implement properly
    set_unique_labels = set(np.unique(labels))
    cv2_video_object = cv2.VideoCapture(path_to_video)
    progress_bar = tqdm(total=int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Writing images to file...')
    # width, height = cv2_video_object.get(3), cv2_video_object.get(4)  # TODO: low: address unused variables
    labels = np.hstack((labels[0], labels))  # fill the first frame  # TODO: why need to fill first frame?
    frame_count = 0  # TODO: med: rename `count` -- what is it counting? Other variable `i` already tracks iterations over the while loop
    i = 0
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)

    while cv2_video_object.isOpened():
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            # Prepare writing info onto image
            text_for_frame = f'Label__'
            # Try appending label
            try:
                label_word = config.map_group_to_behaviour[labels[i]]
                text_for_frame += label_word
            except KeyError:
                text_for_frame += f'NotFound. Group: {labels[i]}'
                label_not_found_err = f'Label number not found: {labels[i]}. Set of unique ' \
                                      f'label numbers = {set_unique_labels}'
                logger.error(label_not_found_err)
            except IndexError as ie:
                index_err = f'Index error. Could not index i ({i}) onto labels (labels.shape={labels.shape}) / ' \
                            f'is_frame_retrieved = {is_frame_retrieved} / ' \
                            f'Original exception: {repr(ie)}'
                logger.error(index_err)
                raise IndexError(index_err)

            text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]
            # TODO: evaluate magic variables RE: text offsetting on images
            text_offset_x, text_offset_y = 50, 50
            box_coordinates = (
                (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
                (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
            )
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img=frame, text=text_for_frame, org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
            # Write to image
            image_name = f'frame_{frame_count+1}.png'
            cv2.imwrite(os.path.join(output_path, image_name), frame)

            # Save & set metrics, prepare for next frame, and update progress bar
            frame_count += frames_to_skip_after_each_write
            cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
            progress_bar.update(frames_to_skip_after_each_write)
            i += 1
        else:  # No more frames left to retrieve. Release object and finish.
            cv2_video_object.release()
            break
    progress_bar.close()
    return



@config.deco__log_entry_exit(logger)
def create_labeled_vid_NEW_ATTEMPT(labels, critical_behaviour_minimum_duration=3, num_randomly_generated_examples=5,
                                   frames_dir=config.FRAMES_OUTPUT_PATH, output_path=config.SHORT_VIDEOS_OUTPUT_PATH,
                                   output_video_fps=config.OUTPUT_VIDEO_FPS) -> None:
    """

    :param labels:
    :param critical_behaviour_minimum_duration:
    :param num_randomly_generated_examples:
    :param frames_dir:
    :param output_path:
    :param output_video_fps:
    :return:
    """
    # Create list of only .png images found in the frames directory
    images: List[str] = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    if len(images) <= 0:
        empty_list_err = f'{inspect.stack()[0][3]}(): Zero .png frames were found in {frames_dir}. ' \
                         f'Could not created labeled video. Exiting early.'  # TODO:
        logger.error(empty_list_err)
        # raise ValueError(empty_list_err)
        return
    likelihoodprocessing.sort_list_nicely_in_place(images)

    # Extract first image in images list. Set dimensions.
    four_character_code = cv2.VideoWriter_fourcc(*'mp4v')
    first_image = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _layers = first_image.shape

    # ?  TODO
    labels_list, idx_list, lengths_list = augmented_runlength_encoding(labels)
    ranges_list, idx2_list = [], []

    # Loop over lengths
    for idx_length_i, length_i in enumerate(lengths_list):
        if length_i >= critical_behaviour_minimum_duration:
            ranges_list.append(range(idx_list[idx_length_i], idx_list[idx_length_i] + length_i))
            idx2_list.append(idx_length_i)

    # Loop over the range generated from the total unique labels available
    for idx_label in tqdm(range(len(np.unique(labels))),
                          desc=f'{inspect.stack()[0][3]}(): Creating video (TODO: update this bar description)'):
        a: List[range] = []  # TODO: low: `a` needs more description
        for idx_range in range(
                len(ranges_list)):  # TODO: low: could making this loop into a comprehension be more readable?
            if labels_list[idx2_list[idx_range]] == idx_label:
                a += [ranges_list[idx_range], ]  # Previously: a.append(ranges[idx_range])
        try:
            random_ranges = random.sample(a, min(len(a), num_randomly_generated_examples))
            for idx_random_range, random_range_i in enumerate(random_ranges):
                grp_images = []
                # Loop over list of randomly generated ranges
                for a_random_range in random_ranges[idx_random_range]:  # TODO: substitute for `random_range_i`?
                    # Aggregate images into a list that correspond to the randomly generated numbers/ranges
                    try:
                        images_of_interest = images[a_random_range]
                    except IndexError as ie:  # Index out of range
                        index_err = f'{inspect.stack()[0][3]}(): Indexing error. ' \
                                    f'The "random range" appears to be out of bounds for the ' \
                                    f'images actually available. Check logic above. ' \
                                    f'Originally, this error failed silently but it should be addressed later. ' \
                                    f'Original error: {repr(ie)}'
                        logger.error(index_err)
                    else:  # Execute else block if no errors
                        grp_images.append(images_of_interest)

                if len(grp_images) > 0:
                    video_name = f'group_{idx_label} -- example_{idx_random_range}.mp4'
                    # Open video writer object
                    video_writer = cv2.VideoWriter(
                        filename=os.path.join(output_path, video_name),
                        fourcc=four_character_code,
                        fps=output_video_fps,
                        # TODO: med: 5 is a magic variable? FPS? #########################################################
                        frameSize=(width, height)
                    )
                    # Loop over all images and write to file with video writer
                    for image in grp_images:
                        video_writer.write(cv2.imread(os.path.join(frames_dir, image)))
                    # Release video writer and continue

                    video_writer.release()
                cv2.destroyAllWindows()

        except Exception as e:  # TODO: low: exception is very general. Address?
            err = f'{inspect.stack()[0][3]}: Unexpected exception occurred when trying to make video. ' \
                  f'Review exception and address/make function more robust instead of failing silently. ' \
                  f'Exception is: {repr(e)}.'
            logger.error(err)
            raise e
    return


########################################################################################################################
def import_vidfolders(folders: List[str], output_path: List[str]):
    """ * legacy *
    Previously called `import_vidfolders()`
    Import multiple folders containing .mp4 files and extract frames from them
    :param folders: list of folder paths
    :param output_path: list, directory to where you want to store extracted vid images in LOCAL_CONFIG
    """
    list_of_lists_of_videos: List[List[str]] = []  # TODO: Q: why does this variable exist? It tracks but does not contribute to anything
    # Loop through folders
    for idx_folder, folder in enumerate(folders):
        videos_list_from_current_folder: List[str] = get_videos_from_folder_in_BASEPATH(folder)
        # Loop through videos
        for idx_video, video in enumerate(videos_list_from_current_folder):
            logger.debug(f'{inspect.stack()[0][3]}():Extracting frames from {video} and appending labels to these images...')
            # Write (something) to disk TODO
            write_annotated_frames_to_disk_from_video(video, output_path)  # TODO: HIGH: missing param `FPS` *** runtime error imminent ********************************************************
            logger.debug(f'Done extracting images and writing labels, from MP4 file {idx_video+1}')
        # After looping through videos, append list of videos from current folder to list of lists because reasons
        list_of_lists_of_videos.append(videos_list_from_current_folder)  # list_of_lists_of_videos.append(videos_list_from_current_folder)
        logger.info(f'Processed {len(videos_list_from_current_folder)} mp4 files from folder: {folder}.')
    return


# @config.deco__log_entry_exit(logger)
def create_labeled_vid(labels, critical_behaviour_minimum_duration=3, num_randomly_generated_examples=5, frame_dir=config.FRAMES_OUTPUT_PATH, output_path=config.SHORT_VIDEOS_OUTPUT_PATH) -> None:
    """
    (Generalized create_labeled_video() function that works between _py, _umap, and _voc submodules)
    TODO: low: purpose
    :param labels: 1D array, labels from training or testing
    :param critical_behaviour_minimum_duration: scalar, minimum duration for random selection of behaviors, default 300ms
    :param num_randomly_generated_examples: scalar, number of randomly generated examples, default 5  # TODO: low: default is actually 3..does counts refer to cv2.VideoWriter pathing?
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    # TODO: med: in the case that
    # Create list of only .png images found in the frames directory
    images: List[str] = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    if len(images) <= 0:
        empty_list_err = f'{inspect.stack()[0][3]}(): Zero .png frames were found in {frame_dir}. ' \
                         f'Could not created labeled video. Exiting early.'  # TODO:
        logger.error(empty_list_err)
        # raise ValueError(empty_list_err)
        return
    likelihoodprocessing.sort_list_nicely_in_place(images)

    # Extract first image in images list. Set dimensions.
    four_character_code = cv2.VideoWriter_fourcc(*'mp4v')
    first_image = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _layers = first_image.shape

    # ?  TODO
    labels_list, idx_list, lengths_list = augmented_runlength_encoding(labels)
    ranges_list, idx2_list = [], []

    # Loop over lengths
    for idx_length_i, length_i in enumerate(lengths_list):
        if length_i >= critical_behaviour_minimum_duration:
            ranges_list.append(range(idx_list[idx_length_i], idx_list[idx_length_i]+length_i))
            idx2_list.append(idx_length_i)

    # Loop over the range generated from the total unique labels available
    for idx_label in tqdm(range(len(np.unique(labels))), desc=f'{inspect.stack()[0][3]}(): Creating video (TODO: update this bar description)'):
        a: List[range] = []  # TODO: low: `a` needs more description
        for idx_range in range(len(ranges_list)):  # TODO: low: could making this loop into a comprehension be more readable?
            if labels_list[idx2_list[idx_range]] == idx_label:
                a += [ranges_list[idx_range], ]  # Previously: a.append(ranges[idx_range])
        try:
            random_ranges = random.sample(a, min(len(a), num_randomly_generated_examples))
            for idx_random_range, random_range_i in enumerate(random_ranges):
                grp_images = []
                # Loop over list of randomly generated ranges
                for a_random_range in random_ranges[idx_random_range]:  # TODO: substitute for `random_range_i`?
                    # Aggregate images into a list that correspond to the randomly generated numbers/ranges
                    try:
                        images_of_interest = images[a_random_range]
                    except IndexError as ie:  # Index out of range
                        # Original implementation allowed this part to fail silently...address this later when there is time!
                        index_err = f'{inspect.stack()[0][3]}(): Indexing error. ' \
                                    f'The "random range" appears to be out of bounds for the ' \
                                    f'images actually available. Check logic above. ' \
                                    f'Originally, this error failed silently but it should be addressed later. ' \
                                    f'Original error: {repr(ie)}'
                        logger.error(index_err)
                    else:  # Execute else block if no errors
                        grp_images.append(images_of_interest)

                if len(grp_images) > 0:
                    video_name = f'group_{idx_label} / example_{idx_random_range}.mp4'
                    # Open video writer object
                    video_writer = cv2.VideoWriter(
                        os.path.join(output_path, video_name),
                        four_character_code,
                        5,  # TODO: med: 5 is a magic variable? FPS? #########################################################
                        (width, height)
                    )
                    # Loop over all images and write to file with video writer
                    for image in grp_images:
                        video_writer.write(cv2.imread(os.path.join(frame_dir, image)))
                    # Release video writer and continue

                    video_writer.release()

                cv2.destroyAllWindows()

        except Exception as e:  # TODO: low: exception is very general. Address?
            err = f'{inspect.stack()[0][3]}: Unexpected exception occurred when trying to make video. ' \
                  f'Review exception and address/make function more robust instead of failing silently. ' \
                  f'Exception is: {repr(e)}.'
            logger.error(err)

            raise e
    return

@config.deco__log_entry_exit(logger)
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
    likelihoodprocessing.sort_list_nicely_in_place(images)
    four_character_code = cv2.VideoWriter_fourcc(*'avc1')  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(video_frames_directory, images[0]))
    height, width, _layers = frame.shape
    ranges, idx2 = [], []
    n, idx, lengths = augmented_runlength_encoding(labels)
    for idx_length, length in enumerate(lengths):
        if length >= crit:
            ranges.append(range(idx[idx_length], idx[idx_length] + length))
            idx2.append(idx_length)

    # Loop over labels
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
    warning = f'Current function: {inspect.stack()[0][3]}(). Instead of calling this, call the two functions ' \
              f'inside explicitly. '
    logger.warning(warning)
    write_annotated_frames_to_disk_from_video(path_to_video, labels, fps, output_path)
    create_labeled_vid(labels, critical_behaviour_minimum_duration=3, num_randomly_generated_examples=5, frame_dir=output_path, output_path=config.SHORTVID_DIR)


###
# Legacy functions (some of which will be deprecated)
def vid2frame(path_to_video: str, labels, fps: int, output_path: str = config.FRAMES_OUTPUT_PATH):
    """ # # # DEPRECATION WARNING # # # """
    replacement_func = write_annotated_frames_to_disk_from_video
    logger.error(f'This function, vid2frame(), will be deprecated shortly. The replacement '
                 f'function is called "{replacement_func.__qualname__}" and aims to make usage more clear and DRY. '
                 f'If you are reading this, this function was kept for backwards compatibility reasons. ' 
                 f'Caller = {inspect.stack()[1][3]}')
    return replacement_func(path_to_video, labels, fps, output_path)


def main(path_to_video, labels, fps, output_path):  # To be deprecated
    """# # # DEPRECATION WARNING # # #"""
    replacement_function = get_frames_from_video_then_create_labeled_video
    logger.error('This function, bsoid.util.videoprocessing.main(), will be deprecated in the future in '
                 'favour of a refactored, more descriptive function. Currently, that function is: '
                 f'{replacement_function.__qualname__}. Caller = {inspect.stack()[1][3]}')
    return replacement_function(path_to_video, labels, fps, output_path)
