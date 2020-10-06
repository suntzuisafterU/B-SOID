"""
train_data_sources
"""
from bhtsne import tsne as TSNE_bhtsne
from sklearn import mixture
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Any, Collection, Dict, List, Optional, Tuple, Union
import inspect
import joblib
import numpy as np
import openTSNE
import os
import pandas as pd
import sys
import time

# TODO: low: resolve bsoid imports better
from bsoid import config, feature_engineering, util

logger = config.initialize_logger(__file__)


### TODOS
# TODO: implement ACTUAL random state s.t. all random state property calls beget a truly random integer

###

class PipelineAttributeHolder:
    """
    Helps obfuscate params from base Pipeline object for API clarity
    Implement setters and getters.
    """
    # Base information
    _name: str = 'DefaultPipelineName'
    _description: str = ''
    _folder_source: str = config.OUTPUT_PATH  # Folder in which this pipeline resides

    data_ext: str = 'csv'

    # Tracking vars
    _is_built = False
    _has_unused_raw_data: bool = False  # TODO: add note here: changes to False when new data added and pipeline needs to be rebuilt
    tsne_source: str = None

    # Sources
    train_data_files_paths: List[str] = []
    predict_data_files_paths: List[str] = []

    # Data
    _dfs_list_raw_data: List[pd.DataFrame] = []  # Raw data frames kept right after read_data() function called

    train_data: List[pd.DataFrame] = []
    test_data: List[pd.DataFrame] = []

    # Model objects
    _scaler = None
    _tsne_obj = None
    _clf_gmm = None
    _clf_svm = None
    _random_state: Optional[int] = None
    average_over_n_frames: int = 3
    tsne_dimensions = 3

    SKLEARN_SVM_PARAMS: dict = {}
    SKLEARN_EMGMM_PARAMS: dict = {}
    SKLEARN_TSNE_PARAMS: dict = {}

    kwargs: dict = {}
    # Properties
    @property
    def name(self): return self._name
    @property
    def description(self): return self._description
    @property
    def file_path(self) -> Optional[str]:  # TODO: low: review
        if not self._folder_source:
            return None
        return os.path.join(self._folder_source, generate_pipeline_filename(self.name))
    @property
    def clf_gmm(self): return self._clf_gmm
    @property
    def clf_svm(self): return self._clf_svm
    @property
    def random_state(self): return self._random_state
    @property
    def has_unused_raw_data(self): return self._has_unused_raw_data
    def get_desc(self) -> str: return self._description

    def set_name(self, name):
        # TODO: will this cause problems later with naming convention?
        if util.io.has_invalid_chars_in_name_for_a_file(name):
            invalid_name_err = f'Invalid chars detected for name: {name}.'
            logger.error(invalid_name_err)
            raise ValueError(invalid_name_err)
        self._name = name

        return self
    def set_desc(self, description):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        if not isinstance(description, str):
            invalid_type_err = f'Invalid type submitted. found: {type(description)} TODO clean up this err message'
            logger.error(invalid_type_err)
            raise TypeError(invalid_type_err)
        self._description = description
        return self


class Pipeline(PipelineAttributeHolder):
    """
    Base pipeline object. It enumerates the basic functions by which each pipeline should adhere.

    Parameters/kwargs
    ----------
    train_data_sources : EITHER List[str] OR str, optional
        Specify a source or sources by which the pipeline reads in training data.
        User must ensure that paths

    tsne_source : str, optional (default: 'sklearn')
        Specify a TSNE implementation.
        Valid TSNE implementations are: {sklearn, bhtsne}.
    """
    # TODO: organize housing variables

    dims_cols_names: List[str] = None
    _dfs_list_raw_data: List[pd.DataFrame] = []
    df_features: pd.DataFrame = None
    df_post_tsne: pd.DataFrame = None

    valid_tsne_sources: set = {'bhtsne', 'sklearn', 'opentsne', }
    cross_val_scores: Collection = None

    def __init__(self, name: str = None, tsne_source=None, data_extension='csv', **kwargs):
        """ Initialize pipeline + config """
        # Pipeline name
        if name is not None and not isinstance(name, str):
            raise TypeError(f'name should be of type str but instead found type {type(name)}')
        elif isinstance(name, str): self.set_name(name)
        if tsne_source is not None:
            if not isinstance(tsne_source, str):
                tsne_type_err = f'TODO bad type for tsne source ({type(tsne_source)}'
                logger.error(tsne_type_err)
                raise TypeError(tsne_type_err)
            if tsne_source not in self.valid_tsne_sources:
                tsne_err = f'TODO: non-implemented tsne source: {tsne_source}'
                logger.error(tsne_err)
                raise ValueError(tsne_err)
            self.tsne_source = tsne_source
        # Validate data extension to be pulled from DLC output. Right now, only CSV and h5 supported by DLC to output.
        if data_extension is not None:
            if not isinstance(data_extension, str):
                data_ext_type_err = f'Invalid type found for data ext: {type(data_extension)} /' \
                                    f'TODO: clean up err message'
                logger.error(data_ext_type_err)
                raise TypeError(data_ext_type_err)
            self.data_ext = data_extension


        # TODO: add kwargs parsing for dimensions
        self.dims_cols_names = [f'dim{d+1}' for d in range(self.tsne_dimensions)]
        #
        #
        # TODO: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ****
        self.kwargs = kwargs
        # Final setup
        self.__read_in_kwargs(**kwargs)
        self.__read_in_config_file_vars()

    def __read_in_kwargs(self, **kwargs):
        """ Reads in kwargs else pull form config file """
        # TODO: add kwargs parsing for averaging over n-frames
        # Read in training data sources
        train_data_sources = kwargs.get('train_data_sources')
        if train_data_sources is not None:
            self.read_in_train_data_source(train_data_sources)

        # Random state
        random_state = kwargs.get('random_state')
        if random_state is not None:
            if not isinstance(random_state, int):
                random_state_type_err = f'Invalid type found for random ' \
                                        f'state: {type(random_state)} (value: {random_state})'
                logger.error(random_state_type_err)
                raise TypeError(random_state_type_err)
            self._random_state = random_state
        else: self._random_state = config.RANDOM_STATE
        return

    def read_in_train_data_source(self, *train_data_args):
        """
        Read in new data. Save to raw data list.
        train_data_args: any number of args. Types submitted expected to be either List[str] or [str]
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        # Type checking first. If *args is None or empty collection, raise error.
        if not train_data_args:
            invalid_args_err = f'Invalid arg(s) submitted. Could not read args(s): {train_data_args}'
            logger.error(invalid_args_err)
            raise ValueError(invalid_args_err)
        # If multiple args submitted, recursively call self in the case that an arg is a list.
        if len(train_data_args) > 1:
            for arg in train_data_args:
                if isinstance(arg, list) or isinstance(arg, tuple):
                    self.read_in_train_data_source(arg)
                else:
                    # Do type checking, path checking, then continue.
                    util.check_arg.ensure_type(arg, str)
                    self.read_in_train_data_source(arg)
            return self

        assert len(train_data_args) == 1, f'Based on recursive structure above, arg SHOULD be collection of 1 item'

        path = train_data_args[0]
        if os.path.isdir(path):  # If path is to a directory: check directory for all file files of valid file ext
            data_sources = [os.path.join(path, x) for x in os.listdir(path)
                            if x.split('.'[-1]) == self.data_ext
                            and os.path.join(path, x) not in self.train_data_files_paths]
            if len(data_sources) <= 0: return self
            for file_path in data_sources:
                df_i = util.io.read_csv(file_path)
                self._dfs_list_raw_data.append(df_i)
                self.train_data_files_paths.append(file_path)
        elif os.path.isfile(path):  # If path is to a file: read in file
            ext = path.split('.')[-1]
            if ext != self.data_ext:
                invalid_data_source_err = f'Invalid data source submitted. Expected data type extension ' \
                                          f'of {self.data_ext} but instead found: {ext} (path={path}).'
                logger.error(invalid_data_source_err)
                raise ValueError(invalid_data_source_err)
            if path in self.train_data_files_paths: return self
            df_file = util.io.read_csv(path)
            self._dfs_list_raw_data.append(df_file)
            self.train_data_files_paths.append(path)
        else:
            unusual_path_err = f'Unusual file/dir path submitted but not found: {path}. Is not a valid ' \
                                 f'file and not a directory.'  # TODO: low: improve error msg clarity
            logger.error(unusual_path_err)
            raise ValueError(unusual_path_err)

        self._has_unused_raw_data = True
        return self
    def read_in_predict_data_source(self):
        raise NotImplementedError(f'TODO')  # TODO: low: mimic the read_train_data func above
    def __read_in_config_file_vars(self):
        """
            Check if config variables are inserted. If they are not manually inserted on
        Pipeline instantiation, insert config vars from config.ini.
        """
        if len(self.SKLEARN_SVM_PARAMS) == 0:
            self.SKLEARN_SVM_PARAMS = config.SVM_PARAMS
        if len(self.SKLEARN_EMGMM_PARAMS) == 0:
            self.SKLEARN_EMGMM_PARAMS = config.EMGMM_PARAMS
        if len(self.SKLEARN_TSNE_PARAMS) == 0:
            self.SKLEARN_TSNE_PARAMS = config.TSNE_SKLEARN_PARAMS

    def read_in_predict_folder_data_file_paths_legacypathing(self) -> List[str]:  # TODO: deprecate/re-work
        self.predict_data_files_paths = predict_data_files_paths = [os.path.join(config.PREDICT_DATA_FOLDER_PATH, x)
                                                                    for x in os.listdir(config.PREDICT_DATA_FOLDER_PATH)
                                                                    if x.split('.')[-1] == self.data_ext].copy()
        if len(predict_data_files_paths) <= 0:
            err = f'Zero csv files found from: {config.PREDICT_DATA_FOLDER_PATH}'
            logger.error(err)
        return predict_data_files_paths

    def read_in_train_folder_data_file_paths_legacypathing(self) -> List[str]:  # TODO: deprecate/re-work
        self.train_data_files_paths = predict_data_files_paths = [os.path.join(config.TRAIN_DATA_FOLDER_PATH, x)
                                                                  for x in os.listdir(config.TRAIN_DATA_FOLDER_PATH)
                                                                  if x.split('.')[-1] == self.data_ext].copy()
        if len(predict_data_files_paths) <= 0:
            err = f'Zero csv files found from {config.TRAIN_DATA_FOLDER_PATH}'
            logger.error(err)
            raise ValueError(err)
        return predict_data_files_paths

    @config.deco__log_entry_exit(logger)
    def scale_data(self, df_data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        # TODO: add arg checking to ensure all features specified are in columns
        self._scaler = StandardScaler()
        self._scaler.fit(df_data[features])
        arr_data_scaled = self._scaler.transform(df_data[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)
        for col in df_data.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df_data[col].values
        return df_scaled_data

    @config.deco__log_entry_exit(logger)
    def train_SVM(self, x_train, y_train, x_test=None) -> Optional[np.ndarray]:
        self._clf_svm = SVC(**self.SKLEARN_SVM_PARAMS)
        self._clf_svm.fit(x_train, y_train)
        if x_test is not None:
            return self.clf_svm.predict(x_test)
        return

    @config.deco__log_entry_exit(logger)
    def train_gmm_and_get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Train GMM. Get associated labels. Save GMM. TODO: elaborate
        :param df:
        :return:
        """
        self._clf_gmm = mixture.GaussianMixture(**self.SKLEARN_EMGMM_PARAMS).fit(df)
        assignments = self.clf_gmm.predict(df)
        return assignments

    @config.deco__log_entry_exit(logger)
    def train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        TODO: elaborate
        TODO: ensure that TSNE obj can be saved and used later for new data? *** Important ***
        :param data:
        :param kwargs:
        :return:
        """
        if self.tsne_source == 'bhtsne':
            arr_result = TSNE_bhtsne(
                data,
                dimensions=self.tsne_dimensions,
                perplexity=np.sqrt(len(self.features_names_7)),
                rand_seed=self.random_state, )
        elif self.tsne_source == 'sklearn':
            # TODO: high: Save the TSNE object
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),  # Perplexity scales with sqrt, power law
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n
                **self.SKLEARN_TSNE_PARAMS,
            ).fit_transform(data[self.features_names_7])
        elif self.tsne_source == 'opentsne':
            self._tsne_obj = openTSNE.TSNE(
                negative_gradient_method='bh',  # TODO: make this a changeable var?
                n_components=self.tsne_dimensions,
                n_iter=config.TSNE_N_ITER,
                n_jobs=10,
                verbose=True)
            tsne_embedding = self._tsne_obj.fit(data[self.features_names_7].values)
            arr_result = tsne_embedding.transform(data[self.features_names_7].values)
        else: raise RuntimeError(f'Invalid TSNE source type fell through the cracks: {self.tsne_source}')
        return arr_result

    def save(self, output_path_dir):  # TODO: alt pathing?
        """
        :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
        """
        if output_path_dir is None: output_path_dir = config.OUTPUT_PATH
        # Check if valid directory
        if not os.path.isdir(output_path_dir):
            raise ValueError(f'Invalid output path dir specified: {output_path_dir}')
        final_out_path = os.path.join(output_path_dir, generate_pipeline_filename(self._name))

        # Check if valid final path to be saved
        if not util.io.is_pathname_valid(final_out_path):
            invalid_path_err = f'Invalid output path save: {final_out_path}'
            logger.error(invalid_path_err)
            raise ValueError(invalid_path_err)
        with open(final_out_path, 'wb') as model_file:
            joblib.dump(self, model_file)
        self._folder_source = output_path_dir
        logger.debug(f'Pipeline ({self.name}) saved to: {final_out_path}')
        return self

    def has_been_previously_saved(self):
        if not self._folder_source: return False
        if not os.path.isfile(os.path.join(self._folder_source, generate_pipeline_filename(self.name))): return False
        return True

    def write_video_frames_to_disk(self, video_to_be_labeled=config.VIDEO_TO_LABEL_PATH):
        labels = list(self.df_post_tsne[self.gmm_assignment_col_name].values)
        labels = [f'label_i={i} // label={l}' for i, l in enumerate(labels)]
        # util.videoprocessing.write_annotated_frames_to_disk_from_video_LEGACY(
        #     video_to_be_labeled,
        #     labels,
        # )
        # TODO: ensure new implementation works!!!!!
        util.videoprocessing.write_annotated_frames_to_disk_from_video_NEW(
            video_to_be_labeled,
            labels,
        )
        return

    def make_video_from_written_frames(self, new_video_name, video_to_be_labeled=config.VIDEO_TO_LABEL_PATH):
        util.videoprocessing.write_video_with_existing_frames(
            video_to_be_labeled,
            config.FRAMES_OUTPUT_PATH,
            new_video_name,
        )
        return

    def make_video(self, video_to_be_labeled=config.VIDEO_TO_LABEL_PATH, output_fps=config.OUTPUT_VIDEO_FPS):
        labels = list(self.df_post_tsne[self.gmm_assignment_col_name].values)
        labels = [f'label={l}' for l in labels]
        if not os.path.isfile(video_to_be_labeled):
            not_a_video_err = f'The video to be labeled is not a valid file/path. ' \
                              f'Submitted video path: {video_to_be_labeled}. '
            logger.error(not_a_video_err)
        else:

            self.write_video_frames_to_disk()
            # Write video
            self.make_video_from_written_frames()

        return self

    def plot_assignments_in_3d(self, show_now=False, save_to_file=False, azim_elev=(70, 135)):
        # TODO: low: check for other runtiem vars
        if not self._is_built:  # or not self._has_unused_raw_data:
            logger.warning(f'Classifiers have not been built. Nothing to graph.')
            return None

        self.fig_gm_assignments_3d = util.visuals.plot_GM_assignments_in_3d(
            self.df_post_tsne[self.dims_cols_names].values,
            self.df_post_tsne[self.gmm_assignment_col_name].values,
            save_to_file,
            show_now=show_now,
            azim_elev=azim_elev,
        )

        return self.fig_gm_assignments_3d

    def plot(self):
        logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        # # plot 3d stuff?
        util.visuals.plot_GM_assignments_in_3d(self.df_post_tsne[self.dims_cols_names].values, self.df_post_tsne[self.gmm_assignment_col_name].values, config.SAVE_GRAPHS_TO_FILE)

        # below plot is for cross-val scores
        scores = cross_val_score()  # TODO: low
        # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')

        # TODO: fix below
        # util.visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
        logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')
        return

    # Misc attributes
    gmm_assignment_col_name = 'gmm_assignment'
    svm_assignment_col_name = 'svm_assignment'

    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength',
                                      'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
    features_which_average_by_sum = ['SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed']
    features_names_7 = features_which_average_by_mean + features_which_average_by_sum


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

    @config.deco__log_entry_exit(logger)
    def engineer_features(self, list_dfs_raw_data: Union[List[pd.DataFrame], pd.DataFrame]) -> Pipeline:
        """
        All functions that take the raw data (data retrieved from using bsoid.read_csv()) and
        transforms it into classifier-ready data.
        :param list_dfs_raw_data: (DataFrame or list of DataFrames)
        :return:
        (Includes scorer, source cols)
        """
        if isinstance(list_dfs_raw_data, pd.DataFrame): list_dfs_raw_data = [list_dfs_raw_data, ]
        elif not isinstance(list_dfs_raw_data, list): raise TypeError(f'Invalid type found: {type(list_dfs_raw_data)}')  # TODO: elaborate

        # Adaptively filter features
        dfs_list_adaptively_filtered: List[Tuple[pd.DataFrame, List[float]]] = \
            [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]

        # Engineer features as necessary
        dfs_features = []
        for df_i, _ in tqdm(dfs_list_adaptively_filtered, desc='Engineer features...'):
            # Save scorer, source values because the current way of engineering features strips out that info.
            scorer, source = df_i['scorer'][0], df_i['source'][0]
            df_features_i = feature_engineering.engineer_7_features_dataframe_NOMISSINGDATA(
                df_i, features_names_7=self.features_names_7)
            if 'scorer' not in df_features_i.columns:
                df_features_i['scorer'] = scorer
            if 'source' not in df_features_i.columns:
                df_features_i['source'] = source
            dfs_features.append(df_features_i)

        # Smooth over n-frame windows
        for i, df in tqdm(enumerate(dfs_features), desc='Smoothing values over frames...'):
            # Mean
            for feature in self.features_which_average_by_mean:
                dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
                    df[feature].values, 'avg', self.average_over_n_frames)
            # Sum
            for feature in self.features_which_average_by_sum:
                dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
                    df[feature].values, 'sum', self.average_over_n_frames)
        # Aggregate all train data
        df_features_all = pd.concat(dfs_features)
        self.df_features = df_features_all

        return self

    def tsne_reduce_df_features(self) -> Pipeline:
        arr_tsne_result = self.train_tsne_get_dimension_reduced_data(self.df_features_scaled)
        self.df_post_tsne = pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names)
        return self

    @config.deco__log_entry_exit(logger)
    def build(self):
        self.test_build()

    def test_build(self) -> Pipeline:
        """
        TODO
        :param save:
        :param inplace:
        :return:
        """

        train_data_files_paths: List[str] = self.read_in_train_folder_data_file_paths_legacypathing()
        train_data_files_paths = self.train_data_files_paths

        # Read in train data
        self.dfs_list_raw_data = dfs_list_raw_data = [util.io.read_csv(file_path)
                                                      for file_path in train_data_files_paths]

        # Engineer features
        logger.debug('Start engineering features')
        start = time.perf_counter()
        self.engineer_features(dfs_list_raw_data)
        logger.debug(f'Finished engineering features in {round(time.perf_counter() - start, 1)} seconds.')

        # Scale data
        self.df_features_scaled = self.scale_data(self.df_features, self.features_names_7)

        # TSNE -- create new dimensionally reduced data
        self.tsne_reduce_df_features()

        # Train GMM, get assignments
        self.df_post_tsne[self.gmm_assignment_col_name] = self.train_gmm_and_get_labels(self.df_post_tsne[self.dims_cols_names])

        # Test-train split
        df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(
            self.df_post_tsne[self.dims_cols_names], self.df_post_tsne[self.gmm_assignment_col_name],
            test_size=config.HOLDOUT_PERCENT, random_state=config.RANDOM_STATE)  # TODO: add shuffle kwarg?
        self.df_features_train, self.df_features_test, self.df_labels_train, self.df_labels_test = df_features_train, df_features_test, df_labels_train, df_labels_test

        # # Train SVM
        df_labels_train[self.svm_assignment_col_name] = self.train_SVM(df_features_train, df_labels_train, df_features_test)
        self.df_labels_train = df_labels_train

        # Get cross-val, accuracy scores
        self.cross_val_scores = cross_val_score(
            self.clf_svm, df_features_test, df_labels_train[self.svm_assignment_col_name])

        self.acc_score = accuracy_score(
            y_pred=self.clf_svm.predict(df_features_test), y_true=df_labels_train[self.svm_assignment_col_name])

        # # Do plotting, save info as necessary
        if config.PLOT_GRAPHS:  # TODO; silently kill this section for now
            logger.debug(f'Enter GRAPH PLOTTING section of {inspect.stack()[0][3]}')
            # util.visuals.plot_GM_assignments_in_3d(df_post_tsne[self.dims_cols_names].values, df_post_tsne[self.gmm_assignment_col_name].values, config.SAVE_GRAPHS_TO_FILE)

            # below plot is for cross-val scores
            # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')

            # TODO: fix below
            # util.visuals.plot_feats_bsoidpy(features_10fps, gmm_assignments)
            logger.debug(f'Exiting GRAPH PLOTTING section of {inspect.stack()[0][3]}')

        self.is_built = True
        logger.debug(f'All done with building classifiers!')
        return self

    def run(self):
        """ Runs after build(). Using terminology from old implementation. TODO """
        # read in paths
        data_files_paths: List[str] = self.read_in_predict_folder_data_file_paths_legacypathing()
        # Read in PREDICT data
        dfs_raw = [util.io.read_csv(csv_path) for csv_path in data_files_paths]

        # Engineer features accordingly (as above)
        df_features = self.engineer_features(dfs_raw)
        # TODO: low
        # Predict labels
            # How does frameshifting 2x fit in?
        # Generate videos

        return self


def generate_pipeline_filename(name):
    file_name = f'{name}.pipeline'
    return file_name


if __name__ == '__main__':
    BSOID = os.path.dirname(os.path.dirname(__file__))
    if BSOID not in sys.path:
        sys.path.insert(0, BSOID)

    run = True
    if run:
        # Test build
        nom = 'TestPipeline43'
        loc = 'C:\\Users\\killian\\Pictures'
        full_loc = os.path.join(loc, f'{nom}.pipeline')
        actual_save_loc = f'C:\\{nom}.pipeline'

        make_new = False
        if make_new:
            p = TestPipeline1(name=nom, tsne_source='sklearn').build()
            p.save(loc)
            print(f'Accuracy score: {p.acc_score}')

        read_existing = True
        if read_existing:
            # p = bsoid.read_pipeline(actual_save_loc)
            p = util.io.read_pipeline(full_loc)
            print(str(p.train_data_files_paths))
            p.plot_assignments_in_3d(show_now=True)

        pass

        """
        
        C:\TestPipeline42.pipeline
        
        """

# C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos