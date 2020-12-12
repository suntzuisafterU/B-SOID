"""

For ever new pipeline implementation by the user, make sure you do the following:
    - Use BasicPipeline as the parent object
    - Implement `engineer_features()` using the given interface


Notes
    - the OpenTSNE implementation does not allow more than 2 components
    - GMM's "reg covar" == "regularization covariance"
TODO:
    med/high: review use of UMAP -- potential alternative to SVC?
    med/high: review use of HDBSCAN -- possible replacement for GMM clustering?
    low: implement ACTUAL random state s.t. all random state property calls beget a truly random integer
    low: review "theta"(/angle) for TSNE

Add attrib checking for engineer_features? https://duckduckgo.com/?t=ffab&q=get+all+classes+within+a+file+python&ia=web&iax=qa


"""
from bhtsne import tsne as TSNE_bhtsne
from pandas.core.common import SettingWithCopyWarning
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle as sklearn_shuffle_dataframe
from typing import Any, Collection, Dict, List, Optional, Tuple, Union  # TODO: med: review all uses of Optional
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import sys
import time
# from tqdm import tqdm
# import openTSNE  # openTSNE only supports n_components 2 or less
# import warnings

from bsoid.logging_bsoid import get_current_function
from bsoid import check_arg, config, feature_engineering, io, statistics, videoprocessing, visuals


logger = config.initialize_logger(__file__)


# Base pipeline objects that outline the API

class PipelineAttributeHolder(object):
    """
    Helps hide params from base Pipeline object for API clarity
    Implement setters and getters.
    """
    # Base information
    _name, _description = 'DefaultPipelineName', '(Default pipeline description)'
    # data_ext: str = 'csv'  # Extension which data is read from  # TODO: deprecate, delete line
    dims_cols_names = None  # Union[List[str], Tuple[str]]
    valid_tsne_sources: set = {'bhtsne', 'sklearn', }
    gmm_assignment_col_name, svm_assignment_col_name,  = 'gmm_assignment', 'svm_assignment'
    behaviour_col_name = 'behaviour'

    # Tracking vars
    _is_built = False  # Is False until the classifiers are built then changes to True

    _is_training_data_set_different_from_model_input: bool = False  # Changes to True if new training data is added and classifiers not rebuilt.
    _has_unengineered_predict_data: bool = False    # Changes to True if new predict data is added. Changes to False if features are engineered.
    _has_modified_model_variables: bool = False

    # Data
    default_cols = ['data_source', 'file_source']  #, svm_assignment_col_name, gmm_assignment_col_name]
    df_features_train_raw = pd.DataFrame(columns=default_cols)
    df_features_train = pd.DataFrame(columns=default_cols)
    df_features_train_scaled = pd.DataFrame(columns=default_cols)
    df_features_predict_raw = pd.DataFrame(columns=default_cols)
    df_features_predict = pd.DataFrame(columns=default_cols)
    df_features_predict_scaled = pd.DataFrame(columns=default_cols)

    # Other model vars (Rename this)
    cross_validation_k: int = config.CROSSVALIDATION_K
    _random_state: int = None
    average_over_n_frames: int = 3  # TODO: low: add to kwargs? Address later.
    test_train_split_pct: float = None

    # Model objects
    _scaler: StandardScaler = None
    _clf_gmm: GaussianMixture = None
    _clf_svm: SVC = None
    # TSNE
    tsne_source: str = 'sklearn'
    tsne_n_components: int = 3
    tsne_n_iter: int = None
    tsne_early_exaggeration: float = None  # Defaults to config.ini value if not specified in kwargs
    tsne_n_jobs: int = None  # n cores used during process
    tsne_verbose: int = None
    # GMM
    gmm_n_components, gmm_covariance_type, gmm_tol, gmm_reg_covar = None,  None, None, None
    gmm_max_iter, gmm_n_init, gmm_init_params = None, None, None
    gmm_verbose: int = None
    gmm_verbose_interval: int = None
    # SVM
    svm_c, svm_gamma, svm_probability, svm_verbose = None, None, None, None

    # Column names
    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength',
                                      'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
    features_which_average_by_sum = ['SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed']
    _all_features: Tuple[str] = tuple(features_which_average_by_mean + features_which_average_by_sum)
    test_col_name = 'is_test_data'

    label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9 = ['' for _ in range(10)]
    label_10, label_11, label_12, label_13, label_14, label_15, label_16, label_17, label_18 = ['' for _ in range(9)]
    label_19, label_20, label_21, label_22, label_23, label_24, label_25, label_26, label_27 = ['' for _ in range(9)]
    label_28, label_29, label_30, label_31, label_32, label_33, label_34, label_35, label_36 = ['' for _ in range(9)]

    # Misc attributes
    kwargs: dict = {}
    _last_built: str = None

    # SORT ME
    _acc_score: float = None
    _cross_val_scores: np.ndarray = np.array([])
    seconds_to_engineer_train_features: float = None

    # TODO: high: evaluate
    def get_assignment_label(self, assignment: int) -> str:
        """ Get behavioural label according to assignment value (number) """
        try:
            assignment = int(assignment)
        except ValueError:
            err = f'TODO: elaborate error: invalid assignment submitted: "{assignment}"'
            logger.error(err)
            raise ValueError(err)

        try:  # This try/catch is a debugging effort. Remove try/catch later when ironed-out.
            label = getattr(self, f'label_{assignment}')
        except Exception as e:
            logger.error(f'{get_current_function()}(): unexpected exception '
                         f'occurred! Please address. Label not found: "label_{assignment}"')
            raise e

        return label

    def set_label(self, assignment, label: str):
        """ Set behavioural label for a given model assignment number/value """
        assignment = int(assignment)
        setattr(self, f'label_{assignment}', label)
        return self

    # # # Getters/Properties
    @property
    def is_in_inconsistent_state(self):
        """
        Useful for checking if training data has been added/removed from pipeline
        relative to already-compiled model
        """
        return self._is_training_data_set_different_from_model_input

    @property
    def name(self): return self._name
    @property
    def description(self): return self._description
    @property
    def clf_gmm(self): return self._clf_gmm
    @property
    def clf_svm(self): return self._clf_svm
    @property
    def random_state(self): return self._random_state
    def get_desc(self) -> str: return self._description
    @property
    def is_built(self): return self._is_built
    @property
    def accuracy_score(self): return self._acc_score
    @property
    def scaler(self): return self._scaler
    @property
    def svm_col(self) -> str: return self.svm_assignment_col_name
    @property
    def svm_assignment(self) -> str: return self.svm_assignment_col_name
    @property
    def cross_val_scores(self):  # Union[List, np.ndarray]
        return self._cross_val_scores
    @property
    def training_data_sources(self) -> List[str]:
        return list(np.unique(self.df_features_train_raw['data_source'].values))
    @property
    def predict_data_sources(self):  # List[str]
        return list(np.unique(self.df_features_predict_raw['data_source'].values))
    @property
    def raw_assignments(self):  # List[str]
        return self.raw_assignments
    @property
    def unique_assignments(self) -> List[any]:
        if len(self.df_features_train_scaled) > 0:
            return list(np.unique(self.df_features_train_scaled[self.svm_col].values))
        return []
    @property
    def all_features(self) -> Collection[str]:
        return self._all_features
    @property
    def all_features_list(self) -> List[str]:
        return list(self._all_features)

    # Setters
    def set_name(self, name: str):
        # TODO: will this cause problems later with naming convention?
        check_arg.ensure_has_valid_chars_for_path(name)
        self._name = name
        return self

    def set_description(self, description):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        check_arg.ensure_type(description, str)
        self._description = description
        return self


class BasePipeline(PipelineAttributeHolder):
    """BasePipeline

    It enumerates the basic functions by which each pipeline should adhere.


    Parameters
    ----------
    name : str
        Name of pipeline. Also is the name of the saved pipeline file.


    kwargs
        Kwargs default to pulling in data from config.ini file unless overtly specified to override. See below.
    ----------

    tsne_source : {'sklearn', 'bhtsne'}
        Specify a TSNE implementation to use for dimensionality reduction.
        Must be one of:

        'sklearn'
            sklearn explanation goes here
        'bhtsne'
            bhtsne explanation goes here

        # TODO: med: expand on further kwargs
    """
    # Init
    def __init__(self, name: str, save_folder: str = None, **kwargs):
        # Pipeline name
        check_arg.ensure_type(name, str)
        self.set_name(name)
        # TSNE source
        tsne_source = kwargs.get('tsne_source', '')
        check_arg.ensure_type(tsne_source, str)
        if tsne_source in self.valid_tsne_sources:
            self.tsne_source = tsne_source
        #
        self.kwargs = kwargs
        # Final setup
        self.set_params(read_config_on_missing_param=True, **kwargs)

    def set_params(self, read_config_on_missing_param=False, **kwargs):
        """
        Reads in variables to change for pipeline.

        If optional arg `read_config_on_missing_param` is True, then any parameter NOT mentioned
        explicitly will be read in from the config.ini file and then replace the current value
        for that property in the pipeline.
        """
        # TODO: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ****
        # TODO: LOW: add kwargs parsing for averaging over n-frames
        # Random state  # TODO: low ensure random state correct
        random_state = kwargs.get('random_state', config.RANDOM_STATE if read_config_on_missing_param else self.random_state)
        check_arg.ensure_type(random_state, int)
        self._random_state = random_state
        ### TSNE ###
        ## SKLEARN ##
        tsne_n_components = kwargs.get('tsne_n_components', config.TSNE_N_COMPONENTS if read_config_on_missing_param else self.tsne_n_components)  # TODO: low: shape up kwarg name for n components? See string name
        check_arg.ensure_type(tsne_n_components, int)
        self.tsne_n_components = tsne_n_components
        tsne_n_iter = kwargs.get('tsne_n_iter', config.TSNE_N_ITER if read_config_on_missing_param else self.tsne_n_iter)
        check_arg.ensure_type(tsne_n_iter, int)
        self.tsne_n_iter = tsne_n_iter
        tsne_early_exaggeration = kwargs.get('tsne_early_exaggeration', config.TSNE_EARLY_EXAGGERATION if read_config_on_missing_param else self.tsne_early_exaggeration)
        check_arg.ensure_type(tsne_early_exaggeration, float)
        self.tsne_early_exaggeration = tsne_early_exaggeration
        n_jobs = kwargs.get('tsne_n_jobs', config.TSNE_N_JOBS if read_config_on_missing_param else self.tsne_n_jobs)
        check_arg.ensure_type(n_jobs, int)
        self.tsne_n_jobs = n_jobs
        tsne_verbose = kwargs.get('tsne_verbose', config.TSNE_VERBOSE if read_config_on_missing_param else self.tsne_verbose)
        check_arg.ensure_type(tsne_verbose, int)
        self.tsne_verbose = tsne_verbose
        # GMM vars
        gmm_n_components = kwargs.get('gmm_n_components', config.gmm_n_components if read_config_on_missing_param else self.gmm_n_components)
        check_arg.ensure_type(gmm_n_components, int)
        self.gmm_n_components = gmm_n_components
        gmm_covariance_type = kwargs.get('gmm_covariance_type', config.gmm_covariance_type if read_config_on_missing_param else self.gmm_covariance_type)
        check_arg.ensure_type(gmm_covariance_type, str)
        self.gmm_covariance_type = gmm_covariance_type
        gmm_tol = kwargs.get('gmm_tol', config.gmm_tol if read_config_on_missing_param else self.gmm_tol)
        check_arg.ensure_type(gmm_tol, float)
        self.gmm_tol = gmm_tol
        gmm_reg_covar = kwargs.get('gmm_reg_covar', config.gmm_reg_covar if read_config_on_missing_param else self.gmm_reg_covar)
        check_arg.ensure_type(gmm_reg_covar, float)
        self.gmm_reg_covar = gmm_reg_covar
        gmm_max_iter = kwargs.get('gmm_max_iter', config.gmm_max_iter if read_config_on_missing_param else self.gmm_max_iter)
        check_arg.ensure_type(gmm_max_iter, int)
        self.gmm_max_iter = gmm_max_iter
        gmm_n_init = kwargs.get('gmm_n_init', config.gmm_n_init if read_config_on_missing_param else self.gmm_n_init)
        check_arg.ensure_type(gmm_n_init, int)
        self.gmm_n_init = gmm_n_init
        gmm_init_params = kwargs.get('gmm_init_params', config.gmm_init_params if read_config_on_missing_param else self.gmm_init_params)
        check_arg.ensure_type(gmm_init_params, str)
        self.gmm_init_params = gmm_init_params
        gmm_verbose = kwargs.get('gmm_verbose', config.gmm_verbose if read_config_on_missing_param else self.gmm_verbose)
        check_arg.ensure_type(gmm_verbose, int)
        self.gmm_verbose = gmm_verbose
        gmm_verbose_interval = kwargs.get('gmm_verbose_interval', config.gmm_verbose_interval if read_config_on_missing_param else self.gmm_verbose_interval)
        check_arg.ensure_type(gmm_verbose_interval, int)
        self.gmm_verbose_interval = gmm_verbose_interval
        # SVM vars
        svm_c = kwargs.get('svm_c', config.svm_c if read_config_on_missing_param else self.svm_c)
        self.svm_c = svm_c
        svm_gamma = kwargs.get('svm_gamma', config.svm_gamma if read_config_on_missing_param else self.svm_gamma)
        self.svm_gamma = svm_gamma
        svm_probability = kwargs.get('svm_probability', config.svm_probability if read_config_on_missing_param else self.svm_probability)
        self.svm_probability = svm_probability
        svm_verbose = kwargs.get('svm_verbose', config.svm_verbose if read_config_on_missing_param else self.svm_verbose)
        self.svm_verbose = svm_verbose
        cross_validation_k = kwargs.get('cross_validation_k', config.CROSSVALIDATION_K if read_config_on_missing_param else self.cross_validation_k)
        check_arg.ensure_type(cross_validation_k, int)
        self.cross_validation_k = cross_validation_k

        # TODO: low/med: add kwargs for parsing test/train split pct
        if self.test_train_split_pct is None:
            self.test_train_split_pct = config.HOLDOUT_PERCENT

        self.dims_cols_names = [f'dim_{d + 1}' for d in range(self.tsne_n_components)]  # TODO: low: encapsulate elsewhere

        self._has_modified_model_variables = True
        return self

    # Functions that should be overwritten by child classes
    def engineer_features(self, data: pd.DataFrame):
        """

        """
        err = f'{get_current_function()}(): Not Implemented for base ' \
              f'Pipeline object {self.__name__}. You must implement this for all child objects.'
        logger.error(err)
        raise NotImplementedError(err)

    # Add & delete data
    def add_train_data_source(self, *train_data_paths_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.
        If the source is a path, then all CSV files directly within that directory will
        be read in as DLC files
        # TODO: add in

        train_data_args: any number of args. Types submitted expected to be of type [str]
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        train_data_paths_args = [path for path in train_data_paths_args
                                 if os.path.split(path)[-1].split('.')[0]  # <- Get file name without extension
                                 not in set(self.df_features_train_raw['data_source'].values)]
        for path in train_data_paths_args:
            if os.path.isfile(path):
                df_new_data = io.read_csv(path)
                self.df_features_train_raw = self.df_features_train_raw.append(df_new_data)
                self._is_training_data_set_different_from_model_input = True
                logger.debug(f'Added file to train data: {path}')
            elif os.path.isdir(path):
                logger.debug(f'Attempting to pull DLC files from {path}')
                data_sources: List[str] = [os.path.join(path, file_name)
                                           for file_name in os.listdir(path)
                                           if file_name.split('.')[-1] in config.valid_dlc_output_extensions
                                           and file_name not in set(self.df_features_train_raw['data_source'].values)]
                for file_path in data_sources:
                    df_new_data_i = io.read_csv(file_path)
                    self.df_features_train_raw = self.df_features_train_raw.append(df_new_data_i)
                    self._is_training_data_set_different_from_model_input = True
                    logger.debug(f'Added file to train data: {file_path}')
            else:
                unusual_path_err = f'Unusual file/dir path submitted but not found: "{path}". Is not a valid ' \
                                   f'file and not a directory.'
                logger.error(unusual_path_err)
                raise ValueError(unusual_path_err)

        return self

    def add_predict_data_source(self, *predict_data_path_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.

        predict_data_args: any number of args. Types submitted expected to be of type str.
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        predict_data_path_args = [path for path in predict_data_path_args
                                  if os.path.split(path)[-1].split('.')[0]  # <- Get file name without extension
                                  not in set(self.df_features_predict_raw['data_source'].values)]
        for path in predict_data_path_args:
            if os.path.isfile(path):
                df_new_data = io.read_csv(path)
                self.df_features_predict_raw = self.df_features_predict_raw.append(df_new_data)
                self._has_unengineered_predict_data = True
                logger.debug(f'Added file to predict data: {path}')
            elif os.path.isdir(path):
                logger.debug(f'Attempting to pull DLC files from {path}')
                data_sources: List[str] = [os.path.join(path, file_name)
                                           for file_name in os.listdir(path)
                                           if file_name.split('.')[-1] in config.valid_dlc_output_extensions
                                           and file_name not in set(self.df_features_predict_raw['data_source'].values)]
                for file_path in data_sources:
                    df_new_data_i = io.read_csv(file_path)
                    self.df_features_predict_raw = self.df_features_predict_raw.append(df_new_data_i)
                    self._has_unengineered_predict_data = True
                    logger.debug(f'Added file to predict data: {file_path}')
            else:
                unusual_path_err = f'Unusual file/dir path submitted but not found: {path}. Is not a valid ' \
                                   f'file and not a directory.'
                logger.error(unusual_path_err)
                raise ValueError(unusual_path_err)

        return self

    def remove_train_data_source(self, data_source: str):
        """"""
        # TODO: low: ensure function, add tests
        check_arg.ensure_type(data_source, str)
        self.df_features_train_raw = self.df_features_train_raw.loc[
            self.df_features_train_raw['data_source'] != data_source]
        self.df_features_train = self.df_features_train.loc[
            self.df_features_train['data_source'] != data_source]
        self.df_features_train_scaled = self.df_features_train_scaled.loc[
            self.df_features_train_scaled['data_source'] != data_source]

        return self

    def remove_predict_data_source(self, data_source: str):
        """
        Remove data from predicted data set.
        :param data_source: (str) name of a data source
        """
        # TODO: low: ensure function, add tests
        check_arg.ensure_type(data_source, str)
        self.df_features_predict_raw = self.df_features_predict_raw.loc[
            self.df_features_predict_raw['data_source'] != data_source]
        self.df_features_predict = self.df_features_predict.loc[
            self.df_features_predict['data_source'] != data_source]
        self.df_features_predict_scaled = self.df_features_predict_scaled.loc[
            self.df_features_predict_scaled['data_source'] != data_source]
        return self

    # Engineer features
    def engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The main function that can build features for BOTH training and prediction data.
        Here we are ensuring that the data processing for both training and prediction occurs in the same way.
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints

        # Reconcile args
        if isinstance(list_dfs_of_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_of_raw_data, ]

        check_arg.ensure_type(list_dfs_of_raw_data, list)

        # Execute
        list_dfs_engineered_features: List[pd.DataFrame] = []
        for i, df in enumerate(list_dfs_of_raw_data):
            df = df.copy().astype({'frame': float})
            check_arg.ensure_frame_indices_are_integers(df)
            logger.debug(f'')
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
            list_dfs_engineered_features.append(df_engineered_features)

        # Aggregate all data into one DataFrame, return
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features

    def engineer_features_train(self):
        """
        Utilizes
        All functions that take the raw data (data retrieved from using bsoid.read_csv()) and
        transforms it into classifier-ready data.

        Post-conditions: sets
        Returns self.
        """
        # TODO: low: save feature engineering time for train data
        start = time.perf_counter()
        # Queue data

        list_dfs_raw_data = [self.df_features_train_raw.loc[self.df_features_train_raw['data_source'] == src]
                                 .astype({'frame': float}).sort_values('frame').copy()
                             for src in set(self.df_features_train_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering training data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering training data features.')
        # Save data
        self.df_features_train = df_features
        # Wrap up
        end = time.perf_counter()
        self._is_training_data_set_different_from_model_input = False
        self.seconds_to_engineer_train_features = round(end-start, 1)
        return self

    def engineer_features_predict(self):
        """ TODO
        """
        # Queue data
        list_dfs_raw_data = [self.df_features_predict_raw.loc[self.df_features_predict_raw['data_source'] == src]
                                 .sort_values('frame').copy()
                             for src in set(self.df_features_predict_raw['data_source'].values)]
        # Call engineering function
        logger.debug(f'Start engineering predict data features.')
        df_features = self.engineer_features_all_dfs(list_dfs_raw_data)
        logger.debug(f'Done engineering predict data features.')
        # Save data, return
        self.df_features_predict = df_features
        self._has_unengineered_predict_data = False
        return self

    ## Scaling data
    def _create_scaled_data(self, df_data, features, create_new_scaler: bool = False) -> pd.DataFrame:
        """
        A universal data scaling function that is usable for training data as well as new prediction data.
        Scales down features in place and does not keep original data.
        """
        # Check args
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_columns_in_DataFrame(df_data, features)
        # Execute
        if create_new_scaler:
            self._scaler = StandardScaler()
            self._scaler.fit(df_data[features])
        arr_data_scaled: np.ndarray = self.scaler.transform(df_data[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)
        # For new DataFrame, replace columns that were not scaled so that data does not go missing
        for col in df_data.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df_data[col].values
        return df_scaled_data

    def scale_transform_train_data(self, features: Collection[str] = None, create_new_scaler=True):
        """
        Scales training data. By default, creates new scaler according to train
        data and stores it in pipeline

        :param features:
        :param create_new_scaler:

        :return: self
        """
        # Queue up data to use
        if features is None:  # TODO: low: remove his if statement as a default feature
            features = self.all_features
        df_features_train = self.df_features_train
        # Check args
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_columns_in_DataFrame(df_features_train, features)
        # Get scaled data
        df_scaled_data = self._create_scaled_data(df_features_train, list(features), create_new_scaler=create_new_scaler)
        check_arg.ensure_type(df_scaled_data, pd.DataFrame)  # Debugging effort. Remove later.
        # Save data. Return.
        self.df_features_train_scaled = df_scaled_data
        return self

    def scale_transform_predict_data(self, features: Optional[List[str]] = None):
        """
        Scales prediction data. Utilizes existing scaler.
        If no feature set is explicitly specified, then the default features set in the Pipeline are used.
        :param features:
        :return:
        """
        # Queue up data to use
        if features is None:
            features = self.all_features
        features = list(features)
        df_features_predict = self.df_features_predict

        # Check args before execution
        check_arg.ensure_type(features, list, tuple)
        check_arg.ensure_type(df_features_predict, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(df_features_predict, features)

        # Get scaled data
        df_scaled_data: pd.DataFrame = self._create_scaled_data(df_features_predict, features, create_new_scaler=False)
        check_arg.ensure_type(df_scaled_data, pd.DataFrame)  # Debugging effort. Remove later.

        # Save data. Return.
        self.df_features_predict_scaled = df_scaled_data
        return self

    # TSNE Transformations
    def train_tsne_get_dimension_reduced_data(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        TODO: elaborate
        TODO: ensure that TSNE obj can be saved and used later for new data? *** Important ***
        :param data:
        :param kwargs:
        :return:
        """
        # Check args
        check_arg.ensure_type(data, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(data, self.all_features_list)
        # Do
        # TODO: uncommment this bhtsne later since it was originally commented-out because VCC distrib. problems
        # if self.tsne_source == 'bhtsne':
        #     arr_result = TSNE_bhtsne(
        #         data[self.features_names_7],
        #         dimensions=self.tsne_n_components,
        #         perplexity=np.sqrt(len(self.features_names_7)),  # TODO: implement math somewhere else
        #         rand_seed=self.random_state,
        #     )
        if False:
            pass
        elif self.tsne_source == 'sklearn':
            # TODO: high: Save the TSNE object
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),  # Perplexity scales with sqrt, power law  # TODO: encapsulate this later
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n  # TODO: encapsulate this later
                n_components=self.tsne_n_components,
                random_state=self.random_state,
                n_iter=self.tsne_n_iter,
                early_exaggeration=self.tsne_early_exaggeration,
                n_jobs=self.tsne_n_jobs,
                verbose=self.tsne_verbose,
            ).fit_transform(data[list(self.all_features_list)])
        else:
            raise RuntimeError(f'Invalid TSNE source type fell through the cracks: {self.tsne_source}')
        return arr_result

    def train_GMM(self, data: pd.DataFrame):
        """"""

        self._clf_gmm = GaussianMixture(
            n_components=self.gmm_n_components,
            covariance_type=self.gmm_covariance_type,
            tol=self.gmm_tol,
            reg_covar=self.gmm_reg_covar,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            init_params=self.gmm_init_params,
            verbose=self.gmm_verbose,
            verbose_interval=self.gmm_verbose_interval,
            random_state=self.random_state,
        ).fit(data)
        return self

    def gmm_predict(self, data):  # TODO: low: remove func?
        assignments = self.clf_gmm.predict(data)
        return assignments

    def train_SVM(self):
        """ Use scaled training data to train SVM classifier """
        df = self.df_features_train_scaled
        # Instantiate SVM object
        self._clf_svm = SVC(
            C=self.svm_c,
            gamma=self.svm_gamma,
            probability=self.svm_probability,
            verbose=self.svm_verbose,
            random_state=self.random_state,
        )
        # Fit SVM to non-test data
        self._clf_svm.fit(
            X=df.loc[~df[self.test_col_name]][list(self.all_features)],  # TODO: too specific
            y=df.loc[~df[self.test_col_name]][self.gmm_assignment_col_name],
        )
        return self
    
    # Higher level data processing functions
    def tsne_reduce_df_features_train(self):
        arr_tsne_result = self.train_tsne_get_dimension_reduced_data(self.df_features_train)
        self.df_features_train_scaled = pd.concat([
            self.df_features_train_scaled,
            pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names),
        ], axis=1)
        return self

    # Model building
    def build_model(self, reengineer_train_features: bool = False):
        """
        Builds the model for predicting behaviours.
        :param reengineer_train_features: (bool) If True, forces the training data to be re-engineered.
        """
        # Engineer features
        logger.debug(f'{inspect.stack()[0][3]}(): Start engineering features')
        if reengineer_train_features or self._is_training_data_set_different_from_model_input:
            self.engineer_features_train()

        # Scale data
        logger.debug(f'Scaling data now...')
        self.scale_transform_train_data(features=self.all_features, create_new_scaler=True)

        # TSNE -- create new dimensionally reduced data
        logger.debug(f'TSNE reducing features now...')
        self.tsne_reduce_df_features_train()

        # Train GMM, get assignments
        logger.debug(f'Training GMM now...')
        self.train_GMM(self.df_features_train_scaled[self.dims_cols_names])
        self.df_features_train_scaled[self.gmm_assignment_col_name] = self.clf_gmm.predict(
            self.df_features_train_scaled[self.dims_cols_names].values)

        # Test-train split
        self.add_test_data_column_to_scaled_train_data()

        # # Train SVM
        logger.debug(f'Training SVM now...')
        self.train_SVM()
        self.df_features_train_scaled[self.svm_assignment_col_name] = self.clf_svm.predict(
            self.df_features_train_scaled[list(self.all_features)].values)
        self.df_features_train_scaled[self.svm_assignment_col_name] = self.df_features_train_scaled[
            self.svm_assignment_col_name].astype(int)

        # # Get cross-val accuracy scores
        self._cross_val_scores = cross_val_score(
            self.clf_svm,
            self.df_features_train_scaled[list(self.all_features)],
            self.df_features_train_scaled[self.svm_assignment_col_name],
            cv=self.cross_validation_k,
        )

        df_features_train_scaled_test_data = self.df_features_train_scaled.loc[~self.df_features_train_scaled[self.test_col_name]]
        self._acc_score = accuracy_score(
            y_pred=self.clf_svm.predict(df_features_train_scaled_test_data[list(self.all_features)]),
            y_true=df_features_train_scaled_test_data[self.svm_assignment_col_name].values)
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')

        # Final touches. Save state of pipeline.
        self._is_built = True
        self._is_training_data_set_different_from_model_input = False  # TODO: review these 3 variables
        self._has_modified_model_variables = False
        self._last_built = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
        logger.debug(f'All done with building classifiers/model!')

        return self

    def build_classifier(self, reengineer_train_features: bool = False):
        """ This is the legacy naming. Method kept for backwards compatibility. This function will be deleted later. """
        return self.build_model(reengineer_train_features=reengineer_train_features)

    def generate_predict_data_assignments(self, reengineer_train_data_features: bool = False, reengineer_predict_features = False):  # TODO: low: rename?
        """ Runs after build(). Using terminology from old implementation. TODO: purpose """
        # TODO: add arg checking for empty predict data?

        # Check that classifiers are built on the training data
        if not self.is_built or reengineer_train_data_features:
            self.build_model()

        # Check if predict features have been engineered
        if reengineer_predict_features or self._has_unengineered_predict_data:
            self.engineer_features_predict()
            self.scale_transform_predict_data()

        # Add prediction labels
        self.df_features_predict_scaled[self.svm_assignment_col_name] = self.clf_svm.predict(
            self.df_features_predict_scaled[list(self.all_features)].values)

        return self

    def build(self, reengineer_train_features=False, reengineer_predict_features=False):
        """
        Build all classifiers and get predictions from predict data
        """
        # Build model
        self.build_model(reengineer_train_features=reengineer_train_features)
        # Get predict data
        self.generate_predict_data_assignments(reengineer_predict_features=reengineer_predict_features)
        return self
    
    # More data transformations
    def add_test_data_column_to_scaled_train_data(self):
        """
        Add boolean column to scaled training data DataFrame to assign train/test data
        """
        test_data_col_name = self.test_col_name
        check_arg.ensure_type(test_data_col_name, str)

        df = self.df_features_train_scaled
        df_shuffled = sklearn_shuffle_dataframe(df)  # Shuffles data, loses none in the process. Assign bool accordingly

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        df_shuffled[test_data_col_name] = False  # TODO: Setting with copy warning occurs on this exact line. is this not how to instantiate it? https://realpython.com/pandas-settingwithcopywarning/
        df_shuffled.loc[:int(len(df) * self.test_train_split_pct), test_data_col_name] = True

        df_shuffled = df_shuffled.reset_index()
        self.df_features_train_scaled = df_shuffled

        return self

    # Saving and stuff
    def save(self, output_path_dir=config.OUTPUT_PATH):
        """
        Defaults to config.ini OUTPUT_PATH variable if a save path not specified beforehand.
        :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
        """
        # if output_path_dir is None:
        #     output_path_dir = config.OUTPUT_PATH
        logger.debug(f'{inspect.stack()[0][3]}(): Attempting to save pipeline to the following folder: {output_path_dir}.')

        # Check if valid directory
        check_arg.ensure_is_dir(output_path_dir)

        # Execute
        final_out_path = os.path.join(output_path_dir, generate_pipeline_filename(self._name))
        # Check if valid final path to be saved
        check_arg.ensure_is_valid_path(final_out_path)
        if not check_arg.is_pathname_valid(final_out_path):  # TODO: low: review
            invalid_path_err = f'Invalid output path save: {final_out_path}'
            logger.error(invalid_path_err)
            raise ValueError(invalid_path_err)

        logger.debug(f'{inspect.stack()[0][3]}(): Attempting to save pipeline to file: {final_out_path}.')

        # Write to file
        try:
            with open(final_out_path, 'wb') as model_file:
                joblib.dump(self, model_file)
        except Exception as e:
            err = f'{get_current_function()}(): An unexpected error occurred: {repr(e)}'
            logger.error(err)
            raise e

        logger.debug(f'{inspect.stack()[0][3]}(): Pipeline ({self.name}) saved to: {final_out_path}')
        return io.read_pipeline(final_out_path)

    def save_to_folder(self, dir: str):
        """  """
        # Arg checking
        if not os.path.isdir(dir):
            not_a_dir_err = f'Argument `dir` = "{dir}" is not a valid directory. Saving pipeline to dir failed.'
            logger.error(not_a_dir_err)
            raise NotADirectoryError(not_a_dir_err)
        # Execute
        # TODO: implement execution
        return

    def save_as(self, file_path):
        """

        :param file_path:
        :return:
        """
        # Arg check
        # TODO: implement arg checking
        # Execute
        # TODO: implement
        return

    # Video stuff

    def make_video(self, video_to_be_labeled: str, data_source: str, video_name: str, output_dir: str, output_fps: int = config.OUTPUT_VIDEO_FPS):
        """

        :param video_to_be_labeled:
        :param output_fps:
        :return:
        """

        # Arg checking
        if not os.path.isfile(video_to_be_labeled):
            not_a_video_err = f'The video to be labeled is not a valid file/path. ' \
                              f'Submitted video path: {video_to_be_labeled}. '
            logger.error(not_a_video_err)
            raise FileNotFoundError(not_a_video_err)
        if not self.is_built:
            err = f'Model is not built so cannot make labeled video'
            logger.error(err)
            raise Exception(err)

        ### Execute
        # Get corresponding data
        if data_source in self.training_data_sources:
            df_data = self.df_features_train_scaled.loc[self.df_features_train_scaled['data_source'] == data_source]
        elif data_source in self.predict_data_sources:
            df_data = self.df_features_predict_scaled.loc[self.df_features_predict_scaled['data_source'] == data_source]
        else:
            err = f'{get_current_function()}(): Data source not found: "{data_source}"'
            logger.error(err)
            raise ValueError(err)

        df_data = df_data.astype({'frame': int}).sort_values('frame')

        svm_assignment_values_array = df_data[self.svm_assignment].values
        labels = list(map(self.get_assignment_label, svm_assignment_values_array))
        frames = list(df_data['frame'].astype(int).values)

        # Generate video with variables
        logger.debug(f'{get_current_function()}(): labels indices example: {labels[:5]}')
        logger.debug(f'Frame indices example: {frames[:5]}')
        videoprocessing.make_labeled_video_according_to_frame(   # labels_list: Union[List, Tuple], frames_indices_list: Union[List, Tuple], output_file_name: str, video_source: str, current_behaviour_list: List[str] = [], output_fps=15, fourcc='H264', output_dir=config.EXAMPLE_VIDEOS_OUTPUT_PATH, **kwargs):
            labels,
            frames,
            video_name,
            video_to_be_labeled,
            output_fps=output_fps,
            output_dir=output_dir,

        )
        # videoprocessing.generate_video_with_labels(
        #     labels,
        #     video_to_be_labeled,
        #     video_name,
        #     output_fps,  # TODO: med/high: magic variable: output fps for video
        # )

        return self

    def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_leadup=0, output_fps=15):
        """
        Create video clips of behaviours

        :param data_source:
        :param video_file_path:
        :param file_name_prefix:
        :param min_rows_of_behaviour:
        :param max_examples:
        :return:
        """
        # TODO: WIP
        text_bgr = (255, 255, 255)
        # Args checking
        check_arg.ensure_type(num_frames_leadup, int)
        check_arg.ensure_is_file(video_file_path)
        # Solve kwargs
        if file_name_prefix is None:
            file_name_prefix = ''
        else:
            check_arg.ensure_type(file_name_prefix, str)
            check_arg.ensure_has_valid_chars_for_path(file_name_prefix)
            file_name_prefix += '__'

        # Get data from data source name
        if data_source in self.training_data_sources:
            df = self.df_features_train_scaled
        elif data_source in self.predict_data_sources:
            df = self.df_features_predict_scaled
        else:
            err = f'Data source not found: {data_source}'
            logger.error(err)
            raise KeyError(err)
        logger.debug(f'{get_current_function()}(): Total records: {len(df)}')

        ### Execute
        # Get DataFrame of the data
        df = df.loc[df["data_source"] == data_source].astype({'frame': int}).sort_values('frame').copy()

        # Get Run-Length Encoding of assignments
        assignments = df[self.svm_assignment_col_name].values
        rle: Tuple[List, List, List] = statistics.augmented_runlength_encoding(assignments)

        # Zip RLE according to order
        # First index is value, second is index, third is additional length
        rle_zipped_by_entry = []
        for row__assignment_idx_addedLength in zip(*rle):
            rle_zipped_by_entry.append(list(row__assignment_idx_addedLength))

        # Roll up assignments into a dict. Keys are labels, values are lists of [index, additional length]
        rle_by_assignment: Dict[Any: List[int, int]] = {}  # Dict[Any: List[int, int]]
        for label, idx, additional_length in rle_zipped_by_entry:
            rle_by_assignment[label] = []
            if additional_length >= min_rows_of_behaviour - 1:
                rle_by_assignment[label].append([idx, additional_length])
        for key in rle_by_assignment.keys():
            rle_by_assignment[key] = sorted(rle_by_assignment[key], key=lambda x: x[1])

        ### Finally: make video clips
        # Loop over assignments
        for key, values_list in rle_by_assignment.items():
            # Loop over examples
            num_examples = min(max_examples, len(values_list))
            # logger.debug(f'')
            for i in range(num_examples):  # TODO: HIGH: this part dumbly loops over first n examples...In the future, it would be better to ensure that at least one of the examples has a long runtime for analysis
                output_file_name = f'{file_name_prefix}{time.strftime("%y-%m-%d_%Hh%Mm")}_' \
                                   f'BehaviourExample__assignment_{key}__example_{i+1}_of_{num_examples}'
                frame_text_prefix = f'Target: {key} / '  # TODO: med/high: magic variable

                idx, additional_length_i = values_list[i]

                lower_bound_row_idx: int = max(0, int(idx) - num_frames_leadup)
                upper_bound_row_idx: int = min(len(df)-1, idx + additional_length_i + 1 + num_frames_leadup)

                df_frames_selection = df.iloc[lower_bound_row_idx:upper_bound_row_idx, :]

                # Compile labels list via SVM assignment for now...Later, we should get the actual behavioural labels instead of the numerical assignments
                labels_list = df_frames_selection[self.svm_assignment_col_name].values
                logger.debug(f'df_frames_selection.dtypes: {df_frames_selection.dtypes}')
                logger.debug(f'df_frames_selection["frame"].dypes.dtypes: {df_frames_selection["frame"].dtypes}')
                frames_indices_list = list(df_frames_selection['frame'].astype(int).values)

                videoprocessing.make_labeled_video_according_to_frame(
                    labels_list,
                    frames_indices_list,
                    output_file_name,
                    video_file_path,
                    text_prefix=frame_text_prefix,
                    text_bgr=text_bgr,
                    output_fps=output_fps,
                )

        return self

    # Diagnostics and graphs
    def get_plot_svm_assignments_distribution(self) -> Tuple[object, object]:
        """
        Get a histogram of assignments in order to review their distribution in the TRAINING data
        """
        fig, ax = visuals.plot_assignment_distribution_histogram(self.df_features_train_scaled[self.svm_assignment_col_name])
        return fig, ax

    def plot_assignments_in_3d(self, show_now=False, save_to_file=False, azim_elev=(70, 135), **kwargs) -> Tuple[object, object]:
        """
        TODO: expand
        :param show_now:
        :param save_to_file:
        :param azim_elev:
        :return:
        """
        # TODO: low: check for other runtime vars
        if not self.is_built:  # or not self._has_unused_raw_data:
            e = f'{get_current_function()}(): The model has not been built. There is nothing to graph.'
            logger.warning(e)
            raise ValueError(e)

        fig, ax = visuals.plot_GM_assignments_in_3d_tuple(
            self.df_features_train_scaled[self.dims_cols_names].values,
            self.df_features_train_scaled[self.gmm_assignment_col_name].values,
            save_to_file,
            show_now=show_now,
            azim_elev=azim_elev,
            **kwargs
        )
        return fig, ax

    def get_plot_cross_val_scoring(self) -> Tuple[object, object]:
        # TODO: med: confirm that this works as expected
        return visuals.plot_cross_validation_scores(self._cross_val_scores)

    def diagnostics(self) -> str:
        """ Function for displaying current state of pipeline. Useful for diagnostics. """
        diag = f"""
self.is_built: {self.is_built}
unique sources in df_train GMM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.gmm_assignment_col_name].values))}
unique sources in df_train SVM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.svm_assignment_col_name].values))}
self._is_training_data_set_different_from_model_input: {self._is_training_data_set_different_from_model_input}
""".strip()
        return diag

    #
    def __repr__(self) -> str:
        # TODO: low: flesh out how these are usually built. Add a last updated info?
        return f'{self.name}'


# Concrete pipeline implementations

class DemoPipeline(BasePipeline):
    """ Demo pipeline used for demonstration on usage. Do not implement this into any real projects. """
    def engineer_features(self, data: pd.DataFrame):
        """ Sample feature engineering function since all
        implementations of BasePipeline must implement this single function. """
        logger.debug(f'Engineering features for one data set...')
        logger.debug(f'Done engineering features.')
        return data


class PipelinePrime(BasePipeline):
    """
    First implementation of a full pipeline. Utilizes the 7 original features from the original B-SOiD paper.

    """
    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered, features_names_7=self.all_features)

        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        return df_features

    def engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The main function that can build features for BOTH training and prediction data.
        Here we are ensuring that the data processing for both training and prediction occurs in the same way.
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints

        list_dfs_raw_data = list_dfs_of_raw_data

        # Reconcile args
        if isinstance(list_dfs_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_raw_data, ]

        check_arg.ensure_type(list_dfs_raw_data, list)

        list_dfs_engineered_features: List[pd.DataFrame] = []
        for df in list_dfs_raw_data:
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
            list_dfs_engineered_features.append(df_engineered_features)


        # # Adaptively filter features
        # dfs_list_adaptively_filtered: List[Tuple[pd.DataFrame, List[float]]] = [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]
        #
        # # Engineer features as necessary
        # dfs_features: List[pd.DataFrame] = []
        # for df_i, _ in tqdm(dfs_list_adaptively_filtered, desc='Engineering features...'):
        #     # Save scorer, source values because the current way of engineering features strips out that info.
        #     df_features_i = feature_engineering.engineer_7_features_dataframe_NOMISSINGDATA(df_i, features_names_7=self.features_names_7)
        #     for col in columns_to_save:
        #         if col not in df_features_i.columns and col in df_i.columns:
        #             df_features_i[col] = df_i[col].values
        #     dfs_features.append(df_features_i)
        #
        # # Smooth over n-frame windows
        # for i, df in tqdm(enumerate(dfs_features), desc='Smoothing values over frames...'):
        #     # Mean
        #     for feature in self.features_which_average_by_mean:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'avg', self.average_over_n_frames)
        #     # Sum
        #     for feature in self.features_which_average_by_sum:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'sum', self.average_over_n_frames)

        # # Aggregate all data
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features


class PipelineEPM(BasePipeline):
    """

    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        check_arg.ensure_type(in_df, pd.DataFrame)
        map_mouse_point_to_config_name = {
            'Head': 'NOSETIP',
            'ForepawLeft': 'FOREPAW_LEFT',
            'ForepawRight': 'FOREPAW_RIGHT',
            'HindpawLeft': 'HINDPAW_LEFT',
            'HindpawRight': 'HINDPAW_RIGHT',
            'Tailbase': 'TAILBASE',
        }

        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered,
            features_names_7=self.features_names_7,
            map_names=map_mouse_point_to_config_name,
        )
        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        # except Exception as e:
        #     logger.error(f'{df_features.columns} // fail on feature: {feature} // {df_features.head(10).to_string()} //{repr(e)}')
        #     raise e

        return df_features


class PipelineFlex(BasePipeline):

    # TODO: WIP: creating a flexible class to use with streamlit that allows for flexible feature selection

    def engineer_features(self, data: pd.DataFrame):
         # TODO
        return data


class PipelineTim(BasePipeline):
    """

    """
    # Feature names
    feat_name_dist_forepawleft_nosetip = 'distForepawLeftNosetip'
    feat_name_dist_forepawright_nosetip = 'distForepawRightNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'distForepawLeftHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'distForepawRightHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'distAvgHindpawNoseTip'
    feat_name_dist_AvgForepaw_NoseTip = 'distAvgForepawNoseTip'
    feat_name_velocity_AvgForepaw = 'velocAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )
    n_rows_to_integrate_by: int = 3  # 3 = 3 frames = 100ms capture in a 30fps video. 100ms was used in original paper.

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_distance_between_2_feats(df, 'FOREPAW_LEFT', 'NOSETIP', self.feat_name_dist_forepawleft_nosetip, resolve_features_with_config_ini=True)
        # 2
        df = feature_engineering.attach_distance_between_2_feats(df, 'FOREPAW_RIGHT', 'NOSETIP', self.feat_name_dist_forepawright_nosetip, resolve_features_with_config_ini=True)
        # 3
        df = feature_engineering.attach_distance_between_2_feats(df, 'FOREPAW_LEFT', 'HINDPAW_LEFT', self.feat_name_dist_forepawLeft_hindpawLeft, resolve_features_with_config_ini=True)
        # 4
        df = feature_engineering.attach_distance_between_2_feats(df, 'FOREPAW_RIGHT', 'HINDPAW_RIGHT', self.feat_name_dist_forepawRight_hindpawRight, resolve_features_with_config_ini=True)
        # 5, 6
        df = feature_engineering.attach_average_forepaw_xy(df)  # TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_hindpaw_xy(df)
        df = feature_engineering.attach_distance_between_2_feats(df, 'AvgHindpaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)
        # 7
        df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        # 8
        df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.n_rows_to_integrate_by)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


### Accessory functions ###

def generate_pipeline_filename(name: str):
    """
    Generates a pipeline file name given its name.

    This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


def generate_pipeline_filename_from_pipeline(pipeline_obj: BasePipeline) -> str:
    return generate_pipeline_filename(pipeline_obj.name)


# Debugging efforts

# if __name__ == '__main__':
#     # This __main__ section is a debugging effort and holds no value to the final product.
#     BSOID = os.path.dirname(os.path.dirname(__file__))
#     if BSOID not in sys.path: sys.path.append(BSOID)
#     test_file_1 = "C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos\\" \
#                   "Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
#     test_file_2 = "C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos\\" \
#                   "Video2DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
#     assert os.path.isfile(test_file_1)
#     assert os.path.isfile(test_file_2)
#
#     run = True
#     if run:
#         # Test build
#         nom = 'videoTest3adsddddddfasdf'
#         loc = 'C:\\Users\\killian\\Pictures'
#         full_loc = os.path.join(loc, f'{nom}.pipeline')
#         actual_save_loc = f'C:\\{nom}.pipeline'
#
#         make_new = True
#         save_new = True
#         if make_new:
#             p = PipelinePrime(name=nom)
#             p = p.add_train_data_source(test_file_1)
#             p = p.add_predict_data_source(test_file_2)
#             p.average_over_n_frames = 5
#             p = p.build_classifier()
#             p = p.generate_predict_data_assignments()
#             if save_new:
#                 p = p.save(loc)
#             print(f'cross_val_scores: {p._cross_val_scores}')
#
#         read_existing = False
#         if read_existing:
#             # p = bsoid.read_pipeline(actual_save_loc)
#             p = io.read_pipeline(full_loc)
#             p.plot_assignments_in_3d(show_now=True)


# if __name__ == '__main__':
#     # This __main__ section is a debugging effort and holds no value to the final product.
#     BSOID = os.path.dirname(os.path.dirname(__file__))
#     if BSOID not in sys.path: sys.path.append(BSOID)
#     test_file_1 = os.path.join(BSOID, 'restraint', 'Restraint-1DLC_resnet50_EPM-RESTRAINTNov2shuffle1_1030000.csv')
#     test_file_2 = os.path.join(BSOID, 'restraint', 'Restraint-2DLC_resnet50_EPM-RESTRAINTNov2shuffle1_1030000.csv')
#
#     assert os.path.isfile(test_file_1), f'path does not exist: {test_file_1}'
#     assert os.path.isfile(test_file_2)
#
#     run = True
#     if run:
#         # # Test build
#         # new_name = 'deleteme'
#         # new_location_to_save_at = 'C:\\Users\\killian\\Pictures'
#         # new_full_loc_to_read = os.path.join(new_location_to_save_at, f'{new_name}.pipeline')
#         # make_new = 0
#         # if make_new:
#         #     save_new = 1
#         #
#         #     p = PipelineTim(name=new_name)
#         #     p = p.add_train_data_source(test_file_1)
#         #     p = p.add_predict_data_source(test_file_2)
#         #     p = p.build()z
#         #     p = p.generate_predict_data_assignments()
#         #     if save_new:
#         #         p = p.save(new_location_to_save_at)
#
#         pipe_in_output = 'colorme.pipeline'
#         pipe_path = os.path.join(config.BSOID_BASE_PROJECT_PATH, 'output', pipe_in_output)
#         data_source = 'Vid3DLC_resnet50_Change_BlindnessNov11shuffle1_850000'
#         vidpath = os.path.join(config.BSOID_BASE_PROJECT_PATH, 'chbo1', 'Vid3.mp4')
#         assert os.path.isfile(pipe_path)
#         assert os.path.isfile(vidpath)
#
#         p = b_io.read_pipeline(pipe_path)
#
#         #     def make_behaviour_example_videos(self, data_source: str, video_file_path: str, file_name_prefix=None, min_rows_of_behaviour=1, max_examples=3, num_frames_leadup=0, output_fps=15):
#         p.make_behaviour_example_videos(
#             data_source,
#             vidpath,
#             file_name_prefix='FirstTryColourMe4',
#             min_rows_of_behaviour=1,
#             max_examples=1,
#             output_fps=0.8,
#             num_frames_leadup=1,
#         )


if __name__ == '__main__':
    # Debugging Objective: ensure that making videos is working well using existing pipeline
    pipe_in_output = 'colorme.pipeline'
    pipe_path = os.path.join(config.BSOID_BASE_PROJECT_PATH, 'output', pipe_in_output)
    data_source = 'Vid3DLC_resnet50_Change_BlindnessNov11shuffle1_850000'
    vidpath = os.path.join(config.BSOID_BASE_PROJECT_PATH, 'chbo1', 'Vid3.mp4')
    assert os.path.isfile(pipe_path)
    assert os.path.isfile(vidpath)

    p = io.read_pipeline(pipe_path)

    p.make_behaviour_example_videos(
        data_source,
        vidpath,
        file_name_prefix='FirstTryColourMe4',
        min_rows_of_behaviour=1,
        max_examples=1,
        output_fps=0.8,
        num_frames_leadup=1,

    )





# streamlit run main.py streamlit -- -p "C:\Users\killian\projects\B-SOID\output\example2.pipeline"
# streamlit run main.py streamlit -- -p "C:\Users\killian\projects\B-SOID\output\example2.pipeline"
# import cv2
# def modify_text_color(frame: np.ndarray, text, font, font_scale, text_offset_x, text_offset_y,text_colour_bgr, rectangle_colour_bgr,  ):
#     text_for_frame = text  # f'{text_prefix}Current Assignment: {label}'
#
#     text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]
#     box_top_left, box_top_right = (
#         (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
#         (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
#     )
#
#     # Add background colour for text
#     cv2.rectangle(frame, box_top_left, box_top_right, rectangle_colour_bgr, cv2.FILLED)
#     # Add text
#     cv2.putText(
#         img=frame,
#         text=str(text_for_frame),
#         org=(text_offset_x, text_offset_y),
#         fontFace=font,
#         fontScale=font_scale,
#         color=text_colour_bgr,
#         thickness=1
#     )

# cd $bsoid ; conda activate bsoid