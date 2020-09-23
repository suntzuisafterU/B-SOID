"""

"""

from sklearn.model_selection import train_test_split, cross_val_score
from typing import Any, List, Tuple
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import time


from bsoid import classify, config, feature_engineering, train, util
from bsoid.config import OUTPUT_PATH, VIDEO_FPS
from bsoid.util.bsoid_logging import get_current_function  # for debugging purposes


###

class TestPipeline(object):

    train_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []

    scaler = None
    clf_svm = None
    clf_em_gmm = None

    def __init__(self):
        pass

    def read_in_all_test_data(self):
        """Read in test data"""
        pass

    def build(self):
        """ Run entire sequence """
        # Read in train data
        # Engineer features as necessary
            # adaptively filter
            # engineer features
            # put into 100 ms bins
        # Aggregate all train data
        # Scale
        # Train TSNE
        # Train GMM
        # Train SVM
        # Do plotting, save info as necesary
        # Save model to file

        pass

    def run(self):
        """ Runs after build(). Using terminology from old implementation """
        # Read in TEST data
        # Engineer features accordingly (as above)
        # Predict labels
            # How does frameshifting 2x fit in?
        # Generate videos



        pass