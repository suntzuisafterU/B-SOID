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
import joblib
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
    """

    """
    data_ext: str = None
    tsne_source: str = None
    valid_tsne_sources: set = {'bhtsne', 'sklearn', }
    train_data_files_paths: List[str] = []
    predict_data_files_paths: List[str] = []

    def __init__(self, tsne_source, data_extension, **kwargs):
        # Validate tsne source type
        if not isinstance(tsne_source, str):
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

    def get_scaled_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        self.scaler = scaler = StandardScaler()
        scaler.fit(df_data)
        arr_data_scaled = scaler.transform(df_data)
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=df_data.columns)
        return df_scaled_data

    def save_pipeline_to_file(self, alternate_output_path=None):  # TODO: alt pathing?
        with open(os.path.join(OUTPUT_PATH, config.PIPELINE_FILENAME), 'wb') as model_file:
            joblib.dump(self, model_file)
        return self

    gmm_assignment_col_name = 'gmm_assignment'
    svm_assignment_col_name = 'svm_assignment'


class TestPipeline(Pipeline):
    """
    Pipelining stuff. TODO.
    Use DataFrames instead of unnamed numpy arrays like the previous iteration
    Parameters
    ----------
    tsne_source : str, optional (default: 'sklearn')
        Specify a TSNE implementation.
        Valid TSNE implementations are: {sklearn, bhtsne}.
    """
    features_names_7 = ['DistFrontPawsTailbaseRelativeBodyLength', 'DistBackPawsBaseTailRelativeBodyLength',
                        'InterforepawDistance', 'BodyLength', 'SnoutToTailbaseChangeInAngle', 'SnoutSpeed',
                        'TailbaseSpeed', ]

    train_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []
    dfs_list_raw_data: List[pd.DataFrame] = []

    data_ext = 'csv'
    tsne_dimensions = 3

    scaler = None
    clf_svm = None
    clf_gmm = None

    def __init__(self, tsne_source: str = 'sklearn', data_ext=data_ext, **kwargs):
        super(TestPipeline, self).__init__(tsne_source=tsne_source, data_extension=data_ext, **kwargs)
        if not isinstance(data_ext, str):
            data_ext_type_err = f'Expected extension for data file to be of type str but ' \
                                f'instead found type: {type(data_ext)}'
            logger.error(data_ext_type_err)
            raise ValueError(data_ext_type_err)
        self.data_ext = data_ext
        self.dims_cols_names = [f'dim{d + 1}' for d in range(self.tsne_dimensions)]
        pass

    def engineer_features(self, df, *args, **kwargs) -> pd.DataFrame:
        return feature_engineering.engineer_7_features_dataframe(df, *args, **kwargs)

    def process_raw_data(self, list_dfs_raw_data, **kwargs) -> pd.DataFrame:
        """  """
        # Adaptively filter features
        dfs__ = [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]
        # Engineer features as necessary
        dfs_features = [self.engineer_features(df_raw_data) for df_raw_data, _ in dfs__]
        # Put into 100ms bins
        # TODO

        # Aggregate all train data
        df_features_all = pd.concat(dfs_features)
        df_features = df_features_all[self.features_names_7]

        return df_features

    def train_tsne(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.tsne_source == 'bhtsne':
            arr_result = TSNE_bhtsne(data,
                                     dimensions=self.tsne_dimensions,
                                     perplexity=np.sqrt(len(self.features_names_7)),
                                     rand_seed=config.RANDOM_STATE, )
        elif self.tsne_source == 'sklearn':
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),  # Perplexity scales with sqrt, power law
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n
                **config.TSNE_SKLEARN_PARAMS,
            ).fit_transform(data[self.features_names_7])
        else: raise RuntimeError(f'Invalid TSNE source type fell through the cracks: {self.tsne_source}')

        return arr_result

    def build(self, inplace=False) -> Pipeline:
        train_data_files_paths: List[str] = self.read_in_train_folder_data_filepaths()
        # Get files names
        train_data_files_paths = [train_data_files_paths[0], ]  # TODO: debugging effort. Delete later.
        # Read in train data
        self.dfs_list_raw_data = dfs_list_raw_data = [util.io.read_csv(file_path)
                                                      for file_path in train_data_files_paths]

        # Engineer features
        df_features = self.process_raw_data(dfs_list_raw_data)
        # Scale data
        df_features_scaled = self.get_scaled_data(df_features[self.features_names_7])

        # TSNE -- create new dimensionally reduced data
        arr_tsne_result = self.train_tsne(df_features_scaled)
        df_post_tsne = pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names)

        # Train GMM
        self.clf_gmm = clf_gmm = mixture.GaussianMixture(**config.EMGMM_PARAMS).fit(df_post_tsne)
        df_post_tsne[self.gmm_assignment_col_name] = clf_gmm.predict(df_post_tsne)

        # # Train SVM
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(
            df_post_tsne[self.dims_cols_names], df_post_tsne[self.gmm_assignment_col_name],
            test_size=config.HOLDOUT_PERCENT, random_state=config.RANDOM_STATE)  # TODO: add shuffle kwarg?
        clf_svm = SVC(**config.SVM_PARAMS)
        clf_svm.fit(df_features_train, df_labels_train)
        df_labels_train[self.svm_assignment_col_name] = clf_svm.predict(df_features_test)

        self.acc_score = accuracy_score(y_pred=clf_svm.predict(df_features_test), y_true=df_labels_train[self.svm_assignment_col_name])

        # # Save model to file
        self.save_pipeline_to_file()

        # # Do plotting, save info as necessary
        if config.PLOT_GRAPHS:
            logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
            util.visuals.plot_classes_EMGMM_assignments(df_post_tsne[self.dims_cols_names].values, df_post_tsne[self.gmm_assignment_col_name].values, config.SAVE_GRAPHS_TO_FILE)

            # below plot is for cross-val scores
            # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')

            # TODO: fix below
            # util.visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
            logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')

        # multi_index_columns = pd.MultiIndex.from_tuples([
        #     ('Features', 'Relative snout to forepaws placement'),
        #     ('', 'Relative snout to hind paws placement'),
        #     ('', 'Inter-forepaw distance'),
        #     ('', 'Body length'),
        #     ('', 'Body angle'),
        #     ('', 'Snout displacement'),
        #     ('', 'Tail-base displacement'),
        #     ('Embedded t-SNE', 'Dimension 1'),
        #     ('', 'Dimension 2'),
        #     ('', 'Dimension 3'),
        #     ('EM-GMM', 'Assignment No.')],
        #     names=['Type', 'Frame@10Hz'])
        # df_training_data = pd.DataFrame(all_data, columns=multi_index_columns)

        # # Write training data to csv
        # training_data_labels_10hz_csv_filename = f'BSOiD__{config.MODEL_NAME}__' \
        #                                          f'train_data_labels__10Hz__{config.runtime_timestr}.csv'
        # df_training_data.to_csv(os.path.join(OUTPUT_PATH, training_data_labels_10hz_csv_filename), index=True, chunksize=10_000, encoding='utf-8')

        return self

    def run(self):
        """ Runs after build(). Using terminology from old implementation. TODO """
        # Read in PREDICT data
        data_files_paths = self.read_in_predict_folder_data_filepaths()

        # Engineer features accordingly (as above)
        # Predict labels
            # How does frameshifting 2x fit in?
        # Generate videos

        return self


if __name__ == '__main__':
    p = TestPipeline().build()
    print(f'Accuracy score: {p.acc_score}')
