"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.

Potential abbreviations:
    sn: snout
    pt: proximal tail ?
"""
from bhtsne import tsne
from sklearn import mixture
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import hdbscan
import inspect
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import umap

from bsoid import config
from bsoid.util import check_arg, likelihoodprocessing, visuals

logger = config.initialize_logger(__name__)


########################################################################################################################

@config.cfig_log_entry_exit(logger)
def extract_7_features_bsoid_tsne_py(list_of_arrays_data: List[np.ndarray], bodyparts=config.BODYPARTS_PY_LEGACY, fps=config.VIDEO_FPS, comp: int = config.COMPILE_CSVS_FOR_TRAINING) -> List[np.ndarray]:
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions

    :param list_of_arrays_data: list of 3D array
    :param bodyparts: dict, body parts with their orders in config
    :param fps: scalar, argument specifying camera frame-rate in config
    :param comp: boolean (0 or 1), argument to compile data or not in config
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    ### *note* Sometimes data is (incorrectly) submitted as an array of arrays (the number of arrays in the overarching array or, if correctly typed, list) is the same # of CSV files read in). Fix type then continue.
    if isinstance(list_of_arrays_data, np.ndarray):
        logger.warning(f'')  # TODO: expand on warning
        list_of_arrays_data = list(list_of_arrays_data)
    # Check args
    check_arg.ensure_type(list_of_arrays_data, list)
    check_arg.ensure_type(list_of_arrays_data[0], np.ndarray)

    ###
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    features: List[np.ndarray] = []

    # Iterate over data arrays available and build features
    for i, data_array in enumerate(list_of_arrays_data):  # for i in range(len(list_of_arrays_data)):
        logger.info(f'Extracting features from CSV file {i+1}...')
        num_data_rows = len(data_array)
        """ DELETE THIS STRING
        7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
            1. body length (or "[d_ST]"): distance from snout to base of tail
            2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT], where [d_FT] is the distance between front paws and base of tail
            3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
            4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

            5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
            6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
            7. snout to base-of-tail change in angle: 

        Author also specifies that: the features are also smoothed over, or averaged across, 
            a sliding window of size equivalent to 60ms (30ms prior to and after the frame of interest).
        """

        # Create some intermediate features first
        inter_forepaw_distance = data_array[:, 2 * bodyparts['Forepaw/Shoulder1']:2 * bodyparts['Forepaw/Shoulder1'] + 2] - data_array[:, 2 * bodyparts['Forepaw/Shoulder2']:2 * bodyparts['Forepaw/Shoulder2'] + 2]  # Originally: 'fpd'

        cfp__center_between_forepaws = np.vstack((
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1']] + data_array[:, 2 * bodyparts['Forepaw/Shoulder2']]) / 2,
            (data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1] + data_array[:, 2 * bodyparts['Forepaw/Shoulder1'] + 1]) / 2),
        ).T  # Originally: cfp
        dFT__cfp_pt__center_between_forepaws__minus__proximal_tail = np.vstack(([
            cfp__center_between_forepaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            cfp__center_between_forepaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: cfp_pt
        chp__center_between_hindpaws = np.vstack((
            ((data_array[:, 2 * bodyparts['Hindpaw/Hip1']] + data_array[:, 2 * bodyparts['Hindpaw/Hip2']]) / 2),
            ((data_array[:, 2 * bodyparts['Hindpaw/Hip1'] + 1] + data_array[:, 2 * bodyparts['Hindpaw/Hip2'] + 1]) / 2),
        )).T
        chp__center_between_hindpaws__minus__proximal_tail = np.vstack(([
            chp__center_between_hindpaws[:, 0] - data_array[:, 2 * bodyparts['Tailbase']],
            chp__center_between_hindpaws[:, 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: chp_pt
        snout__proximal_tail__distance__aka_BODYLENGTH = np.vstack(([
            data_array[:, 2 * bodyparts['Snout/Head']] - data_array[:, 2 * bodyparts['Tailbase']],
            data_array[:, 2 * bodyparts['Snout/Head'] + 1] - data_array[:, 2 * bodyparts['Tailbase'] + 1],
        ])).T  # Originally: sn_pt

        ### Create the 4 static measurement features for final use ###
        inter_forepaw_distance__normalized = np.zeros(num_data_rows)  # Originally: fpd_norm
        cfp_pt__center_between_forepaws__minus__proximal_tail__normalized = np.zeros(num_data_rows)  # Originally: cfp_pt_norm
        chp__proximal_tail__normalized = np.zeros(num_data_rows)  # Originally: chp_pt_norm
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized = np.zeros(num_data_rows)  # Originally: sn_pt_norm
        for j in range(1, num_data_rows):
            inter_forepaw_distance__normalized[j] = np.array(np.linalg.norm(inter_forepaw_distance[j, :]))
            cfp_pt__center_between_forepaws__minus__proximal_tail__normalized[j] = np.linalg.norm(
                dFT__cfp_pt__center_between_forepaws__minus__proximal_tail[j, :])
            chp__proximal_tail__normalized[j] = np.linalg.norm(chp__center_between_hindpaws__minus__proximal_tail[j, :])
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized[j] = np.linalg.norm(
                snout__proximal_tail__distance__aka_BODYLENGTH[j, :])
        ## "Smooth" features for final use
        # Body length (1)
        snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized, win_len)  # Originally: sn_pt_norm_smth
        # Inter-forepaw distance (4)
        inter_forepaw_distance__normalized__smoothed = likelihoodprocessing.boxcar_center(inter_forepaw_distance__normalized, win_len)  # Originally: fpd_norm_smth
        # (2)
        snout__center_forepaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - cfp_pt__center_between_forepaws__minus__proximal_tail__normalized, win_len)  # Originally: sn_cfp_norm_smth
        # (3)
        snout__center_hindpaws__normalized__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__distance__aka_BODYLENGTH__normalized - chp__proximal_tail__normalized, win_len)  # Originally: sn_chp_norm_smth

        ### Create the 3 time-varying features for final use ###
        snout__proximal_tail__angle = np.zeros(num_data_rows - 1)  # Originally: sn_pt_ang
        snout_speed__aka_snout__displacement = np.zeros(num_data_rows - 1)  # Originally: sn_disp
        tail_speed__aka_proximal_tail__displacement = np.zeros(num_data_rows - 1)  # Originally: pt_disp
        for k in range(num_data_rows - 1):
            b_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :], 0])
            a_3d = np.hstack([snout__proximal_tail__distance__aka_BODYLENGTH[k, :], 0])
            c = np.cross(b_3d, a_3d)
            snout__proximal_tail__angle[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi, math.atan2(np.linalg.norm(c), np.dot(snout__proximal_tail__distance__aka_BODYLENGTH[k, :], snout__proximal_tail__distance__aka_BODYLENGTH[k + 1, :])))
            snout_speed__aka_snout__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1] - data_array[k, 2 * bodyparts['Snout/Head']:2 * bodyparts['Snout/Head'] + 1])
            tail_speed__aka_proximal_tail__displacement[k] = np.linalg.norm(data_array[k + 1, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1] - data_array[k, 2 * bodyparts['Tailbase']:2 * bodyparts['Tailbase'] + 1])
        # Smooth features
        snout__proximal_tail__angle__smoothed = likelihoodprocessing.boxcar_center(snout__proximal_tail__angle, win_len)  # Originally: sn_pt_ang_smth
        snout_speed__aka_snout_displacement_smoothed = likelihoodprocessing.boxcar_center(snout_speed__aka_snout__displacement, win_len)  # Originally: sn_disp_smth
        tail_speed__aka_proximal_tail__displacement__smoothed = likelihoodprocessing.boxcar_center(tail_speed__aka_proximal_tail__displacement, win_len)  # Originally: pt_disp_smth

        # Append final features to features list
        features.append(np.vstack((
            snout__center_forepaws__normalized__smoothed[1:],                           # 2
            snout__center_hindpaws__normalized__smoothed[1:],                           # 3
            inter_forepaw_distance__normalized__smoothed[1:],                           # 4
            snout__proximal_tail__distance__aka_BODYLENGTH__normalized_smoothed[1:],    # 1
            # time-varying features
            snout__proximal_tail__angle__smoothed[:],                                   # 7
            snout_speed__aka_snout_displacement_smoothed[:],                            # 5
            tail_speed__aka_proximal_tail__displacement__smoothed[:],                   # 6
        )))
        # End loop // Loop to next data_array
    # Exit
    logger.info(f'{inspect.stack()[0][3]}: Done extracting features from a '
                f'total of {len(list_of_arrays_data)} training CSV files.')

    return features



