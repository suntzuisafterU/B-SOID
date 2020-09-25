"""

"""
from bhtsne import tsne as TSNE_bhtsne
from sklearn import mixture
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import hdbscan
import inspect
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import umap

from bsoid import classify, config, feature_engineering, train, util
from bsoid.config import OUTPUT_PATH, VIDEO_FPS
from bsoid.util.bsoid_logging import get_current_function  # for debugging purposes

logger = config.initialize_logger(__file__)


###

class Pipeline(object):

    data_ext: str = None
    tsne_source: str = None
    valid_tsne_sources = {'bhtsne', 'sklearn', }

    def __init__(self, tsne_source, data_extension, **kwargs):
        if not isinstance(tsne_source, str):
            tsne_type_err = f'TODO bad type for tsne source ({type(tsne_source)}'
            logger.error(tsne_type_err)
            raise TypeError(tsne_type_err)
        if tsne_source not in self.valid_tsne_sources:
            tsne_err = f'TODO: non-implemeneted tsne source: {tsne_source}'
            logger.error(tsne_err)
            raise ValueError(tsne_err)
        self.tsne_source = tsne_source

        self.data_ext = data_extension

        self.kwargs = kwargs

    gmm_assignment_col_name = 'gmm_assignment'
    svm_assignment_col_name = 'svm_assignment'


class TestPipeline(Pipeline):
    """
    Pipelining stuff. TODO.

    Parameters
    ----------
    tsne_source : str, optional (default: 'sklearn')
        Specify a TSNE implementation.
        Valid TSNE implementations are: {sklearn, bhtsne}.
    """

    train_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []
    data_files_paths = []

    data_ext = 'csv'
    tsne_dimensions = 3

    scaler = None
    clf_svm = None
    clf_em_gmm = None

    def __init__(self, tsne_source: str = 'sklearn', data_ext=data_ext, **kwargs):
        super(TestPipeline, self).__init__(tsne_source=tsne_source, data_extension=data_ext, **kwargs)
        if not isinstance(data_ext, str):
            data_ext_type_err = f'Expected extension for data file to be of type str but ' \
                                f'instead found type: {type(data_ext)}'
            logger.error(data_ext_type_err)
            raise ValueError(data_ext_type_err)
        self.data_ext = data_ext
        pass

    def engineer_features(self, df, *args, **kwargs):
        return feature_engineering.engineer_7_features_dataframe(df, *args, **kwargs)

    def read_in_all_test_data(self):
        """Read in test data"""
        # TODO
        return self

    def build(self, inplace=False):
        feats = []
        features_names_7 = ['DistFrontPawsTailbaseRelativeBodyLength', 'DistBackPawsBaseTailRelativeBodyLength',
                            'InterforepawDistance', 'BodyLength', 'SnoutToTailbaseChangeInAngle', 'SnoutSpeed',
                            'TailbaseSpeed', ]
        self.gmm_assignment_label = gmm_assignment_label = 'gmm_assignment'  # TODO: move into class?
        self.svm_assignment_label = svm_assignment_label = 'svm_assignment'
        # Get files names
        self.data_files_paths = data_files_paths = [os.path.join(config.TRAIN_DATA_FOLDER_PATH, x) for x in os.listdir(config.TRAIN_DATA_FOLDER_PATH) if x.split('.')[-1] == self.data_ext]
        data_files_paths = [data_files_paths[0], ]  # TODO: debugging effort. Delete later.
        # Read in train data
        self.dfs_list_raw_data = dfs_list_raw_data = [util.io.read_csv(file_path) for file_path in data_files_paths]
        # Adaptively filter features
        dfs__ = [feature_engineering.adaptively_filter_dlc_output(df) for df in dfs_list_raw_data]
        # Engineer features as necessary
        dfs_features = [self.engineer_features(df_raw_data) for df_raw_data, _ in dfs__]
        # Put into 100ms bins
        # TODO
        # Aggregate all train data
        df_features_all = pd.concat(dfs_features)
        df_features = df_features_all[features_names_7]
        # # Scale
        dims_cols_names = [f'dim{d+1}' for d in range(self.tsne_dimensions)]
        self.scaler = scaler_obj = StandardScaler()
        scaler_obj.fit(df_features)
        df_features_scaled = pd.DataFrame(scaler_obj.transform(df_features), columns=df_features.columns)

        # # Train TSNE
        # TSNE
        # TSNE_sklearn
        if self.tsne_source == 'bhtsne':
            arr_tsne_result = TSNE_bhtsne(df_features_scaled,
                                          dimensions=self.tsne_dimensions,
                                          perplexity=np.sqrt(len(features_names_7)),
                                          rand_seed=config.RANDOM_STATE, )
        elif self.tsne_source == 'sklearn':
            arr_tsne_result = TSNE_sklearn(
                perplexity=np.sqrt(len(df_features_scaled.columns)),  # Perplexity scales with sqrt, power law
                early_exaggeration=16,  # early exaggeration alpha 16 is good
                learning_rate=max(200, len(df_features_scaled.columns) // 16),  # alpha*eta = n
                **config.TSNE_PARAMS,
                n_iter=250,
            ).fit_transform(df_features_scaled[features_names_7])
        else: raise RuntimeError(f'Invalid TSNE source type fell thru the cracks: {self.tsne_source}')

        df_post_tsne = pd.DataFrame(arr_tsne_result, columns=dims_cols_names)

        # Train GMM
        clf_gmm = mixture.GaussianMixture(**config.EMGMM_PARAMS).fit(df_post_tsne)
        df_post_tsne[gmm_assignment_label] = clf_gmm.predict(df_post_tsne)

        # # Train SVM
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(df_post_tsne[dims_cols_names], df_post_tsne[gmm_assignment_label], test_size=config.HOLDOUT_PERCENT, random_state=config.RANDOM_STATE)  # TODO: add shuffle kwarg?
        clf_svm = SVC(**config.SVM_PARAMS)
        clf_svm.fit(df_features_train, df_labels_train)
        df_labels_train[svm_assignment_label] = clf_svm.predict(df_features_test)
        # # Do plotting, save info as necessary
        self.acc_score = acc_score = accuracy_score(clf_svm.predict(df_features_test), df_labels_train[svm_assignment_label])

        # # Save model to file

        return self

    def run(self):
        """ Runs after build(). Using terminology from old implementation """
        # Read in PREDICT data
        # Engineer features accordingly (as above)
        # Predict labels
            # How does frameshifting 2x fit in?
        # Generate videos



        return self


if __name__ == '__main__':
    p = TestPipeline().build()
