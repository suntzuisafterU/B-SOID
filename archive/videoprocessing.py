
from typing import List, Tuple
import cv2
import numpy as np

def label_frame(frame, label):
    """
    A first attempt at encapsulating the "labeling" process (adding text to a
    specified video frame). So far, it hasn't been confirmed to work.
    """
    # TODO: high: evaluate if returning the frame is necessary...don't all CV2 changes occur in place?
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr_black = (0, 0, 0)
    color_white_bgr = (255, 255, 255)

    text_width, text_height = 10, 10  #cv2.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]
    # TODO: evaluate magic variables RE: text offsetting on images
    text_offset_x, text_offset_y = 50, 50
    box_coordinates_topleft, box_coordinates_bottom_right = (
        (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
        (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
    )
    cv2.rectangle(frame, box_coordinates_topleft, box_coordinates_bottom_right, rectangle_bgr_black, cv2.FILLED)
    cv2.putText(img=frame, text=label, org=(text_offset_x, text_offset_y),
                fontFace=font, fontScale=font_scale, color=color_white_bgr, thickness=1)
    return frame


def get_frames_from_video(path_to_video, frames_to_skip_after_each_write=1, **kwargs) -> List[np.ndarray]:
    """
    For a given video, retrieve all specified video frames and return as a list of those frames
    """
    frames = []
    # TODO: resolve fails

    # Arg checking
    assert os.path.isfile(path_to_video), f'Video does not exist: {path_to_video}'
    # Do
    cv2_video_object = cv2.VideoCapture(path_to_video)
    total_frames = int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f'Total frames: {total_frames}')  # Debugging
    i_frame, frame_count = 0, 0
    frames_queue: List[Tuple[bool, Any, str, int]] = []
    # queue up images to write
    done = False
    # Iterate over frames extracted from video. Generate a queue of frames to be labeled later.
    while not done:
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            frames.append(frame)
            # Save & set metrics, prepare for next frame, and update progress bar
            frame_count += frames_to_skip_after_each_write
            # Skip ahead to a specified frame?
            cv2_video_object.set(1, frame_count)
            # progress_bar.update(frames_to_skip_after_each_write)
            i_frame += 1
        else:  # No more frames left to retrieve. Release object and finish.
            done = True
            cv2_video_object.release()
            break

    cv2_video_object.release()
    return frames

