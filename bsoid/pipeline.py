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
from typing import Any, Dict, List, Optional, Tuple, Union
import hdbscan
import inspect
import joblib
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
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
    """

    """
    # TODO: organize housing variables

    train_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []

    dfs_list_raw_data: List[pd.DataFrame] = []
    average_over_n_frames = 3
    data_ext = 'csv'

    scaler = None
    tsne = None
    clf_gmm = None
    clf_svm = None

    tsne_source: str = None
    tsne_dimensions = 3
    valid_tsne_sources: set = {'bhtsne', 'sklearn', 'opentsne'}
    train_data_files_paths: List[str] = []
    predict_data_files_paths: List[str] = []

    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength',
                                      'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
    features_which_average_by_sum = ['SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed']
    features_names_7 = features_which_average_by_mean + features_which_average_by_sum
    gmm_assignment_col_name = 'gmm_assignment'
    svm_assignment_col_name = 'svm_assignment'

    def __init__(self, data_source: str = None, tsne_source=None, data_extension='csv', **kwargs):

        if data_source is not None:
            if not isinstance(data_source, str):
                # TODO: finish err reporting
                data_type_err = f'invalid type: {type(data_source)}'
                raise TypeError(data_type_err)
            if not os.path.isfile(data_source):
                # TODO: finish err reporting
                not_a_file_err = f'Not a file found: {data_source}'
                logger.error(not_a_file_err)
                raise ValueError(not_a_file_err)
            # TOOD: save source?
        # Validate tsne source type
        if tsne_source is not None and not isinstance(tsne_source, str):
            tsne_type_err = f'TODO bad type for tsne source ({type(tsne_source)}'
            logger.error(tsne_type_err)
            raise TypeError(tsne_type_err)
        if tsne_source not in self.valid_tsne_sources:
            tsne_err = f'TODO: non-implemeneted tsne source: {tsne_source}'
            logger.error(tsne_err)
            raise ValueError(tsne_err)
        self.tsne_source = tsne_source
        # Validate data extension to be pulled from DLC output. Right now, only CSV and h5 supported by DLC to output.
        self.data_ext = data_extension
        #
        self.kwargs = kwargs

    def read_in_predict_folder_data_filepaths(self) -> List[str]:
        self.predict_data_files_paths = predict_data_files_paths = [os.path.join(config.PREDICT_DATA_FOLDER_PATH, x)
                                                                    for x in os.listdir(config.PREDICT_DATA_FOLDER_PATH)
                                                                    if x.split('.')[-1] == self.data_ext]
        return predict_data_files_paths

    def read_in_train_folder_data_filepaths(self) -> List[str]:
        self.train_data_files_paths = predict_data_files_paths = [os.path.join(config.TRAIN_DATA_FOLDER_PATH, x)
                                                                  for x in os.listdir(config.TRAIN_DATA_FOLDER_PATH)
                                                                  if x.split('.')[-1] == self.data_ext]
        return predict_data_files_paths

    def scale_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        self.scaler = scaler = StandardScaler()
        scaler.fit(df_data)
        arr_data_scaled = scaler.transform(df_data)
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=df_data.columns)
        return df_scaled_data

    def train_SVM(self, x_train, y_train, x_test=None) -> Optional[np.ndarray]:
        self.clf_svm = SVC(**config.SVM_PARAMS)
        self.clf_svm.fit(x_train, y_train)
        if x_test is not None:
            return self.clf_svm.predict(x_test)
        return

    def train_gmm_and_get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Train GMM. Get associated labels. Save GMM. TODO: elaborate
        :param df:
        :return:
        """
        self.clf_gmm = mixture.GaussianMixture(**config.EMGMM_PARAMS).fit(df)
        assignments = self.clf_gmm.predict(df)
        return assignments

    def train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        TODO: elaborate
        TODO: ensure that TSNE obj can be saved and used on
        :param data:
        :param kwargs:
        :return:
        """
        if self.tsne_source == 'bhtsne':
            arr_result = TSNE_bhtsne(
                data,
                dimensions=self.tsne_dimensions,
                perplexity=np.sqrt(len(self.features_names_7)),
                rand_seed=config.RANDOM_STATE, )
        elif self.tsne_source == 'sklearn':
            # TODO: high: Save the TSNE object
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),  # Perplexity scales with sqrt, power law
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n
                **config.TSNE_SKLEARN_PARAMS,
            ).fit_transform(data[self.features_names_7])
        elif self.tsne_source == 'opentsne':
            self.tsne = openTSNE.TSNE(
                negative_gradient_method='bh',  # TODO: make this a changeable var?
                n_components=self.tsne_dimensions,
                n_iter=config.TSNE_N_ITER,
                n_jobs=10,
                verbose=True)
            tsne_embedding = self.tsne.fit(data[self.features_names_7].values)
            arr_result = tsne_embedding.transform(data[self.features_names_7].values)
        else: raise RuntimeError(f'Invalid TSNE source type fell through the cracks: {self.tsne_source}')
        return arr_result

    def save(self, alternate_output_path=None):  # TODO: alt pathing?
        with open(os.path.join(OUTPUT_PATH, config.PIPELINE_FILENAME), 'wb') as model_file:
            joblib.dump(self, model_file)
        return self

    def build(self):
        raise NotImplementedError(f'build() needs to be implemented for all child Pipeline classes.')


class TestPipeline1(Pipeline):
    """
    Pipelining stuff. TODO.
    Use DataFrames instead of unnamed numpy arrays like the previous iteration
    Parameters
    ----------
    tsne_source : str, optional (default: 'sklearn')
        Specify a TSNE implementation.
        Valid TSNE implementations are: {sklearn, bhtsne}.
    """

    def __init__(self, data_source: str = None, tsne_source: str = 'sklearn', data_ext=None, **kwargs):
        super(TestPipeline1, self).__init__(data_source=data_source, tsne_source=tsne_source, data_extension=data_ext, **kwargs)
        if not isinstance(data_ext, str):
            data_ext_type_err = f'Expected extension for data file to be of type str but ' \
                                f'instead found type: {type(data_ext)}'
            logger.error(data_ext_type_err)
            raise ValueError(data_ext_type_err)
        self.data_ext = data_ext
        self.dims_cols_names = [f'dim{d + 1}' for d in range(self.tsne_dimensions)]

    def engineer_features(self, list_dfs_raw_data: Union[List[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """
        All functions that take the raw data (data retrieved from using bsoid.read_csv()) and
        transforms it into classifier-ready data.
        :param list_dfs_raw_data: (DataFrame or list of DataFrames)
        :return:
        """
        if isinstance(list_dfs_raw_data, pd.DataFrame): list_dfs_raw_data = [list_dfs_raw_data, ]
        elif not isinstance(list_dfs_raw_data, list): raise TypeError(f'Invalid type found: {type(list_dfs_raw_data)}')  # TODO: elaborate

        # Adaptively filter features
        dfs_list_adaptively_filtered = [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]
        # Engineer features as necessary
        dfs_features: List[pd.DataFrame] = [
            feature_engineering.engineer_7_features_dataframe(df_raw_data, features_names_7=self.features_names_7)
            for df_raw_data, _ in dfs_list_adaptively_filtered]
        # Smooth over n-frame windows
        for i, df in enumerate(dfs_features):
            for feature in self.features_which_average_by_mean:
                dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
                    df[feature].values, 'avg', self.average_over_n_frames)
            for feature in self.features_which_average_by_sum:
                dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
                    df[feature].values, 'sum', self.average_over_n_frames)
        # Aggregate all train data
        df_features_all = pd.concat(dfs_features)
        df_features = df_features_all[self.features_names_7]

        return df_features

    def build(self, save = False, inplace=False) -> Pipeline:
        train_data_files_paths: List[str] = self.read_in_train_folder_data_filepaths()
        # Get files names
        train_data_files_paths = [train_data_files_paths[0], ]  # TODO: debugging effort. Delete later.
        # Read in train data
        self.dfs_list_raw_data = dfs_list_raw_data = [util.io.read_csv(file_path)
                                                      for file_path in train_data_files_paths]

        # Engineer features
        df_features = self.engineer_features(dfs_list_raw_data)

        # Scale data
        df_features_scaled = self.scale_data(df_features[self.features_names_7])

        # TSNE -- create new dimensionally reduced data
        arr_tsne_result = self.train_tsne_get_dimension_reduced_data(df_features_scaled)
        df_post_tsne = pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names)

        # Train GMM, get assignments
        df_post_tsne[self.gmm_assignment_col_name] = self.train_gmm_and_get_labels(df_post_tsne)

        # # Train SVM
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(
            df_post_tsne[self.dims_cols_names], df_post_tsne[self.gmm_assignment_col_name],
            test_size=config.HOLDOUT_PERCENT, random_state=config.RANDOM_STATE)  # TODO: add shuffle kwarg?

        df_labels_train[self.svm_assignment_col_name] = self.train_SVM(
            df_features_train, df_labels_train, df_features_test)

        cross_val_scores = cross_val_score(self.clf_svm, df_features_test, df_labels_train[self.svm_assignment_col_name])

        self.acc_score = accuracy_score(y_pred=clf_svm.predict(df_features_test), y_true=df_labels_train[self.svm_assignment_col_name])

        # # Save model to file
        if save:
            self.save()

        # # Do plotting, save info as necessary
        if config.PLOT_GRAPHS and False:  # TODO; silently kill this section for now
            logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
            util.visuals.plot_classes_EMGMM_assignments(df_post_tsne[self.dims_cols_names].values, df_post_tsne[self.gmm_assignment_col_name].values, config.SAVE_GRAPHS_TO_FILE)

            # below plot is for cross-val scores
            # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')

            # TODO: fix below
            # util.visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
            logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')

        return self

    def plot(self):
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        util.visuals.plot_classes_EMGMM_assignments(df_post_tsne[self.dims_cols_names].values, df_post_tsne[self.gmm_assignment_col_name].values, config.SAVE_GRAPHS_TO_FILE)

        # below plot is for cross-val scores
        # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')

        # TODO: fix below
        # util.visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        return

    def run(self):
        """ Runs after build(). Using terminology from old implementation. TODO """
        # read in paths
        data_files_paths: List[str] = self.read_in_predict_folder_data_filepaths()
        # Read in PREDICT data
        dfs_raw = [util.io.read_csv(csv_path) for csv_path in data_files_paths]
        # Engineer features accordingly (as above)
        df_features = self.engineer_features(dfs_raw)
        # Predict labels
            # How does frameshifting 2x fit in?
        # Generate videos

        return self


if __name__ == '__main__':
    p = TestPipeline1(tsne_source='sklearn')\
        .build()
    print(f'Accuracy score: {p.acc_score}')
