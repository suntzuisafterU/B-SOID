"""


Notes
    - the OpenTSNE implementation does not allow more than 2 components
TODOs:
    low: implement ACTUAL random state s.t. all random state property calls beget a truly random integer

"""
from bhtsne import tsne as TSNE_bhtsne
from sklearn.manifold import TSNE as TSNE_sklearn
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score  # , train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Any, Collection, List, Optional, Tuple, Union
import inspect
import joblib
import numpy as np
# import openTSNE  # openTSNE only supports n_components 2 or less
import os
import pandas as pd
import sys
import time


from bsoid import check_arg, config, feature_engineering, io, logging_bsoid, videoprocessing, visuals


logger = config.initialize_logger(__file__)


# Base pipeline objects that outline the API

class PipelineAttributeHolder:
    """
    Helps hide params from base Pipeline object for API clarity
    Implement setters and getters.
    """
    # Base information
    _name, _description = 'DefaultPipelineName', 'Default pipeline description'
    _source_folder: str = config.OUTPUT_PATH  # Folder in which this pipeline resides. OUTPUT_PATH is default.
    data_ext: str = 'csv'  # Extension which data is read from
    dims_cols_names: Union[List[str], Tuple[str]] = None
    valid_tsne_sources: set = {'bhtsne', 'sklearn', 'opentsne', }

    # Tracking vars
    _is_built = False  # Is False until the classifiers are built then changes to True
    _has_unengineered_training_data: bool = False   # Changes to True if new training data
                                                    # is added and classifiers not rebuilt.
    _has_unengineered_predict_data: bool = False    # Changes to True if new predict data
                                                    # is added. Changes to False if features are engineered.
    tsne_source: str = None

    # Sources
    train_data_files_paths: List[str] = []
    predict_data_files_paths: List[str] = []

    # Data
    _dfs_list_raw_train_data: List[pd.DataFrame] = []  # Raw data frames kept right after read_data() function called
    _dfs_list_raw_predict_data: List[pd.DataFrame] = []

    df_features_train: pd.DataFrame = None
    df_features_train_scaled: pd.DataFrame = None
    df_features_predict: pd.DataFrame = None
    df_features_predict_scaled: pd.DataFrame = None

    # Test/train split data
    features_train, features_test, labels_train, labels_test = None, None, None, None

    # Model objects
    _scaler = None
    _clf_gmm = None
    _clf_svm = None

    _random_state: Optional[int] = None
    average_over_n_frames: int = 3  # TODO: low: add to kwargs
    test_train_split_pct: float = None

    # TSNE
    tsne_dimensions: int = None  # Functions in as `n_components` for final dims reducedTODO: low: add to kwargs
    tsne_n_iter: int = None
    tsne_early_exaggeration: float = None
    tsne_n_jobs: int = None  # n cores used during process
    tsne_verbose: int = None
    # GMM
    gmm_n_components: int = None
    gmm_covariance_type = None
    gmm_tol: float = None
    gmm_reg_covar = None
    gmm_max_iter = None
    gmm_n_init = None
    gmm_init_params = None
    gmm_verbose = None
    # SVM
    svm_c: float = None
    svm_gamma: float = None
    svm_probability: bool = None
    svm_verbose: int = None

    # Misc attributes
    gmm_assignment_col_name = 'gmm_assignment'
    svm_assignment_col_name = 'svm_assignment'
    features_which_average_by_mean = ['DistFrontPawsTailbaseRelativeBodyLength',
                                      'DistBackPawsBaseTailRelativeBodyLength', 'InterforepawDistance', 'BodyLength', ]
    features_which_average_by_sum = ['SnoutToTailbaseChangeInAngle', 'SnoutSpeed', 'TailbaseSpeed']
    features_names_7 = features_which_average_by_mean + features_which_average_by_sum
    test_col_name = 'is_test_data'

    #
    kwargs: dict = {}

    # SORT ME
    _acc_score: float = None
    cross_val_scores: Collection[float] = None

    # Properties
    @property
    def name(self): return self._name
    @property
    def description(self): return self._description
    @property
    def file_path(self) -> Optional[str]:  # TODO: low: review
        if not self._source_folder: return None
        return os.path.join(self._source_folder, generate_pipeline_filename(self.name))
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
    def location(self): return self._source_folder
    @property
    def scaler(self): return self._scaler
    # Setters
    def set_name(self, name: str):
        # TODO: will this cause problems later with naming convention?
        if check_arg.has_invalid_chars_in_name_for_a_file(name):
            invalid_name_err = f'Invalid chars detected for name: {name}. Cannot save name.'
            logger.error(invalid_name_err)
            raise ValueError(invalid_name_err)
        self._name = name

        return self
    def set_description(self, description):
        """ Set a description of the pipeline. Include any notes you want to keep regarding the process used. """
        check_arg.ensure_type(description, str)
        self._description = description
        return self
    def set_save_location(self, dir_path):
        raise NotImplementedError(f'change not-saved variable in obj?')
        check_arg.ensure_is_dir(dir_path)
        self._source_folder = dir_path


class BasePipeline(PipelineAttributeHolder):
    """BasePipeline

    It enumerates the basic functions by which each pipeline should adhere.


    Parameters
    ----------
    name : str
        Name of pipeline. Also is the name of the saved pipeline file.


    kwargs
        Kwargs default to pulling in data from config file unless overtly specified to override. See below specs.
    ----------

    data_extension : str
        TODO: low: elaboration

    train_data_source : EITHER List[str] OR str, optional
        Specify a source or sources by which the pipeline reads in training data.
        User must ensure that paths

    tsne_source : {'sklearn', 'bhtsne'}
        Specify a TSNE implementation to use for dimensionality reduction.
        Must be one of:

        'sklearn'
            sklearn explanation goes here
        'bhtsne'
            bhtsne explanation goes here



    """
    # Init
    def __init__(self, name: str = None, save_folder: str = None, tsne_source=None, data_extension='csv', **kwargs):
        # Pipeline name
        if name is not None and not isinstance(name, str):
            raise TypeError(f'name should be of type str but instead found type {type(name)}')
        elif isinstance(name, str):
            self.set_name(name)
        # Save folder
        if save_folder is not None:
            check_arg.ensure_type(save_folder, str)
            if save_folder:
                check_arg.ensure_is_dir(save_folder)
                self._source_folder = save_folder
        # TSNE source
        if tsne_source is not None:
            if not isinstance(tsne_source, str):
                tsne_type_err = f'TODO: LOW: ELABORATE: bad type for tsne source ({type(tsne_source)}'
                logger.error(tsne_type_err)
                raise TypeError(tsne_type_err)
            if tsne_source not in self.valid_tsne_sources:
                tsne_err = f'TODO: LOW: ELABORATE: non-implemented tsne source: {tsne_source}'
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

        # Final setup
        self.__read_in_kwargs(**kwargs)
        self.__read_in_config_file_vars()
        self.dims_cols_names = [f'dim{d + 1}' for d in range(self.tsne_dimensions)]  # TODO
        self.kwargs = kwargs

    def __read_in_kwargs(self, **kwargs):
        """ Reads in kwargs else pull form config file """
        # TODO: ADD KWARGS OPTION FOR OVERRIDING VERBOSE in CONFIG.INI!!!!!!!! ****
        # TODO: LOW: add kwargs parsing for averaging over n-frames
        # TODO: low/med: add kwargs for parsing test/train split pct
        # Read in training data source
        train_data_source = kwargs.get('train_data_source')
        if train_data_source is not None: self.add_train_data_source(train_data_source)
        # Random state  # TODO: low ensure random state correct
        random_state = kwargs.get('random_state', config.RANDOM_STATE)
        if random_state is not None:
            check_arg.ensure_type(random_state, int)
            self._random_state = random_state
        ### TSNE ###
        ## SKLEARN ##
        tsne_n_components = kwargs.get('n_components', config.TSNE_N_COMPONENTS)  # TODO: low: shape up kwarg name for n components? See string name
        check_arg.ensure_type(tsne_n_components, int)
        self.tsne_dimensions = tsne_n_components
        tsne_n_iter = kwargs.get('tsne_n_iter', config.TSNE_N_ITER)
        check_arg.ensure_type(tsne_n_iter, int)
        self.tsne_n_iter = tsne_n_iter
        tsne_early_exaggeration = kwargs.get('tsne_early_exaggeration', config.TSNE_EARLY_EXAGGERATION)
        check_arg.ensure_type(tsne_early_exaggeration, float)
        self.tsne_early_exaggeration = tsne_early_exaggeration
        n_jobs = kwargs.get('tsne_n_jobs', config.TSNE_N_JOBS)
        check_arg.ensure_type(n_jobs, int)
        self.tsne_n_jobs = n_jobs
        tsne_verbose = kwargs.get('tsne_verbose', config.TSNE_VERBOSE)
        check_arg.ensure_type(tsne_verbose, int)
        self.tsne_verbose = tsne_verbose
        # GMM vars
        gmm_n_components = kwargs.get('gmm_n_components', config.gmm_n_components)
        check_arg.ensure_type(gmm_n_components, int)
        self.gmm_n_components = gmm_n_components
        gmm_covariance_type = kwargs.get('gmm_covariance_type', config.gmm_covariance_type)
        check_arg.ensure_type(gmm_covariance_type, str)
        self.gmm_covariance_type = gmm_covariance_type
        gmm_tol = kwargs.get('gmm_tol', config.gmm_tol)
        check_arg.ensure_type(gmm_tol, float)
        self.gmm_tol = gmm_tol
        gmm_reg_covar = kwargs.get('gmm_reg_covar', config.gmm_reg_covar)
        check_arg.ensure_type(gmm_reg_covar, float)
        self.gmm_reg_covar = gmm_reg_covar
        gmm_max_iter = kwargs.get('gmm_max_iter', config.gmm_max_iter)
        check_arg.ensure_type(gmm_max_iter, int)
        self.gmm_max_iter = gmm_max_iter
        gmm_n_init = kwargs.get('gmm_n_init', config.gmm_n_init)
        check_arg.ensure_type(gmm_n_init, int)
        self.gmm_n_init = gmm_n_init
        gmm_init_params = kwargs.get('gmm_init_params', config.gmm_init_params)
        check_arg.ensure_type(gmm_init_params, str)
        self.gmm_init_params = gmm_init_params
        gmm_verbose = kwargs.get('gmm_verbose', config.gmm_verbose)
        check_arg.ensure_type(gmm_verbose, int)
        self.gmm_verbose = gmm_verbose
        # SVM vars
        svm_c = kwargs.get('svm_c', config.svm_c)
        self.svm_c = svm_c
        svm_gamma = kwargs.get('svm_gamma', config.svm_gamma)
        self.svm_gamma = svm_gamma
        svm_probability = kwargs.get('svm_probability', config.svm_probability)
        self.svm_probability = svm_probability
        svm_verbose = kwargs.get('svm_verbose', config.svm_verbose)
        self.svm_verbose = svm_verbose

    def __read_in_config_file_vars(self):
        """
            Check if config variables are inserted. If they are not manually inserted on
        Pipeline instantiation, insert config vars from config.ini.
        """
        if self._source_folder is None:
            self._source_folder = config.OUTPUT_PATH
        if self.test_train_split_pct is None:
            self.test_train_split_pct = config.HOLDOUT_PERCENT
        if len(self.SKLEARN_TSNE_PARAMS) == 0:
            self.SKLEARN_TSNE_PARAMS = config.TSNE_SKLEARN_PARAMS
        if len(self.SKLEARN_EMGMM_PARAMS) == 0:
            self.SKLEARN_EMGMM_PARAMS = config.EMGMM_PARAMS
        if len(self.SKLEARN_SVM_PARAMS) == 0:
            self.SKLEARN_SVM_PARAMS = config.SVM_PARAMS
    # Read/delete data
    def add_train_data_source(self, *train_data_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.
        train_data_args: any number of args. Types submitted expected to be either List[str] or [str]
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        # Type-checking first. If *args is None or empty collection, raise error.
        if not train_data_args:
            # logger.warning(f'Trying to add in an invalid set of items: {train_data_args}')
            return self

        path = train_data_args[0]
        if os.path.isdir(path):  # If path is to a directory: check directory for all file files of valid file ext
            data_sources = [os.path.join(path, x) for x in os.listdir(path)
                            if x.split('.')[-1] == self.data_ext
                            and os.path.join(path, x) not in self.train_data_files_paths]
            for file_path in data_sources:
                df_i = io.read_csv(file_path)
                self._dfs_list_raw_train_data.append(df_i)
                self.train_data_files_paths.append(file_path)
            return self.add_train_data_source(*train_data_args[1:])
        elif os.path.isfile(path):  # If path is to a file: read in file
            ext = path.split('.')[-1]
            if ext != self.data_ext:
                invalid_data_source_err = f'Invalid data source submitted. Expected data type extension ' \
                                          f'of {self.data_ext} but instead found: {ext} (path={path}).'
                logger.error(invalid_data_source_err)
                raise ValueError(invalid_data_source_err)
            if path in self.train_data_files_paths:  #deleteme: add_train_data_source
                return self
            df_file = io.read_csv(path)
            self._dfs_list_raw_train_data.append(df_file)
            self.train_data_files_paths.append(path)
            return self.add_train_data_source(*train_data_args[1:])

        else:
            unusual_path_err = f'Unusual file/dir path submitted but not found: {path}. Is not a valid ' \
                               f'file and not a directory.'  # TODO: low: improve error msg clarity
            logger.error(unusual_path_err)
            raise ValueError(unusual_path_err)
    def add_predict_data_source(self, *predict_data_args):
        """
        Reads in new data and saves raw data. A source can be either a directory or a file.
        train_data_args: any number of args. Types submitted expected to be either List[str] or [str]
            In the case that an arg is List[str], the arg must be a list of strings that
        """
        # Type-checking first. If *args is None or empty collection, raise error.
        if not predict_data_args:
            logger.warning(f'Trying to add in an invalid set of itemsm: {predict_data_args}')
            return self

        path = predict_data_args[0]
        if os.path.isdir(path):  # If path is to a directory: check directory for all file files of valid file ext
            data_sources = [os.path.join(path, x) for x in os.listdir(path)
                            if x.split('.')[-1] == self.data_ext
                            and os.path.join(path, x) not in self.predict_data_files_paths]  # self.train_data_files_paths
            for file_path in data_sources:
                df_i = io.read_csv(file_path)
                self._dfs_list_raw_predict_data.append(df_i)
                self.predict_data_files_paths.append(file_path)
                logger.debug(f'Added file to predict data: {file_path}')
            self._has_unengineered_predict_data = True
            return self.add_predict_data_source(*predict_data_args[1:])
        elif os.path.isfile(path):  # If path is to a file: read in file
            ext = path.split('.')[-1]
            if ext != self.data_ext:
                invalid_data_source_err = f'Invalid data source submitted. Expected data type extension ' \
                                          f'of {self.data_ext} but instead found: {ext} (path={path}).'
                logger.error(invalid_data_source_err)
                raise ValueError(invalid_data_source_err)
            if path in self.predict_data_files_paths:
                return self
            df_file = io.read_csv(path)
            self._dfs_list_raw_predict_data.append(df_file)
            self.predict_data_files_paths.append(path)
            self._has_unengineered_predict_data = True
            logger.debug(f'Added file to predict data: {path}')

            return self.add_predict_data_source(*predict_data_args[1:])

        else:
            unusual_path_err = f'Unusual file/dir path submitted but not found: {path}. Is not a valid ' \
                               f'file and not a directory.'  # TODO: low: improve error msg clarity
            logger.error(unusual_path_err)
            raise ValueError(unusual_path_err)
    def remove_predict_data_source(self, source: str): raise NotImplementedError(f'TODO: Implement')
    def remove_train_data_source(self, source: str): raise NotImplementedError(f'TODO: implement')
    ## Scaling data
    def _create_scaled_data(self, df, features, create_new_scaler: bool = False) -> pd.DataFrame:
        """
        A universal data scaling function that is usable for training data as well as new prediction data.
        Scales down features in place and does not keep original data.
        """
        # TODO: add arg checking to ensure all features specified are in columns
        # Queue up data to use

        # Check args
        check_arg.ensure_type(features, list)
        check_arg.ensure_columns_in_DataFrame(df, features)
        # Do
        if create_new_scaler:
            self._scaler = StandardScaler()
            self._scaler.fit(df[features])
        arr_data_scaled: np.ndarray = self.scaler.transform(df[features])
        df_scaled_data = pd.DataFrame(arr_data_scaled, columns=features)
        # For new DataFrame, replace columns that were not scaled so that data does not go missing
        for col in df.columns:
            if col not in set(df_scaled_data.columns):
                df_scaled_data[col] = df[col].values
        return df_scaled_data
    def scale_transform_train_data(self, features: Optional[List[str]] = None, create_new_scaler=True):
        """
        Scales training data. By default, creates new scaler according to train
        data and stores it in pipeline

        :param features:
        :param create_new_scaler:
        :return:
        """
        # TODO: add arg checking to ensure all features specified are in columns
        # Queue up data to use
        if features is None: features = self.features_names_7
        df_features_train = self.df_features_train
        # Check args
        check_arg.ensure_type(features, list)
        check_arg.ensure_columns_in_DataFrame(df_features_train, features)
        # Get scaled data
        df_scaled_data = self._create_scaled_data(
            df_features_train, features, create_new_scaler=create_new_scaler)
        check_arg.ensure_type(df_scaled_data, pd.DataFrame)  # Debugging effort. Remove later.
        # Save data. Return.
        self.df_features_train_scaled = df_scaled_data
        return self
    def scale_transform_predict_data(self, features: Optional[List[str]] = None):
        """
        Scales prediction data. Utilizes existing scaler.

        :param features:
        :return:
        """
        # Queue up data to use
        if features is None: features = self.features_names_7
        df_features_predict = self.df_features_predict
        # Check args
        check_arg.ensure_type(features, list)
        check_arg.ensure_type(df_features_predict, pd.DataFrame)
        check_arg.ensure_columns_in_DataFrame(df_features_predict, features)
        # Get scaled data
        df_scaled_data: pd.DataFrame = self._create_scaled_data(df_features_predict, features)
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
        #
        if self.tsne_source == 'bhtsne':
            arr_result = TSNE_bhtsne(
                data[self.features_names_7],
                dimensions=self.tsne_dimensions,
                perplexity=np.sqrt(len(self.features_names_7)),  # TODO: implement math somewhere else
                rand_seed=self.random_state,
            )
        elif self.tsne_source == 'sklearn':
            # TODO: high: Save the TSNE object
            arr_result = TSNE_sklearn(
                perplexity=np.sqrt(len(data.columns)),  # Perplexity scales with sqrt, power law  # TODO: encapsulate this later
                learning_rate=max(200, len(data.columns) // 16),  # alpha*eta = n  # TODO: encapsulate this later
                n_components=self.tsne_dimensions,
                random_state=self.random_state,
                n_iter=self.tsne_n_iter,
                early_exaggeration=self.tsne_early_exaggeration,
                n_jobs=self.tsne_n_jobs,
                verbose=self.tsne_verbose,
            ).fit_transform(data[self.features_names_7])
        # elif self.tsne_source == 'opentsne':
        #     tsne_obj = openTSNE.TSNE(
        #         # negative_gradient_method='bh',  # TODO: low: make this a changeable var/add to kwargs
        #         n_components=self.tsne_dimensions,
        #         n_iter=self.tsne_n_iter,
        #         n_jobs=self.tsne_n_jobs,  # TODO: low: magic variable
        #         verbose=bool(self.tsne_verbose),
        #     )
        #     tsne_embedding = tsne_obj.fit(data[self.features_names_7].values)
        #     arr_result = tsne_embedding.transform(data[self.features_names_7].values)
        else:
            raise RuntimeError(f'Invalid TSNE source type fell through the cracks: {self.tsne_source}')
        return arr_result
    def train_GMM(self, data: pd.DataFrame):
        """ TODO: low """
        self._clf_gmm = GaussianMixture(
            n_components=self.gmm_n_components,
            covariance_type=self.gmm_covariance_type,
            tol=self.gmm_tol,
            reg_covar=self.gmm_reg_covar,
            max_iter=self.gmm_max_iter,
            n_init=self.gmm_n_init,
            init_params=self.gmm_init_params,
            verbose=self.gmm_verbose,
            random_state=self.random_state,
        ).fit(data)
        return self
    @config.deco__log_entry_exit(logger)
    def train_gmm_and_get_labels(self, df: pd.DataFrame) -> np.ndarray:  # TODO: remove func?
        """
        Train GMM. Get associated labels. Save GMM. TODO: elaborate
        :param df:
        :return:
        """
        self.train_GMM(df)
        assignments = self.clf_gmm.predict(df)
        return assignments
    def gmm_predict(self, data) -> Any:  # TODO: remove func?
        assignment = self.clf_gmm.predict(data)
        return assignment
    @config.deco__log_entry_exit(logger)
    def train_SVM(self):
        """ TODO; low """
        df = self.df_features_train_scaled
        self._clf_svm = SVC(
            C=self.svm_c,
            gamma=self.svm_gamma,
            probability=self.svm_probability,
            verbose=self.svm_verbose,
            random_state=self.random_state,
            
            # **self.SKLEARN_SVM_PARAMS
        )  # TODO: HIGH: manually input SVM params instead of dict
        self._clf_svm.fit(
            X=df.loc[~df[self.test_col_name]][self.features_names_7],
            y=df.loc[~df[self.test_col_name]][self.gmm_assignment_col_name],
        )

        return self
    # More data transformations
    def engineer_features(self, data: pd.DataFrame):
        err = f'{logging_bsoid.get_current_function()}(): Not Implemented for base Pipeline object.'
        logger.error(err)
        raise NotImplementedError(err)
    def add_test_data_column_to_scaled_train_data(self, test_data_col_name: str = None):
        """
        Add boolean column to scaled training data DataFrame to assign train/test data
        """
        if test_data_col_name is None:
            test_data_col_name = self.test_col_name
        check_arg.ensure_type(test_data_col_name, str)

        df = self.df_features_train_scaled
        df_shuffled = shuffle(df)  # Shuffles data, loses none in the process. Assign bool accordingly.
        df_shuffled[test_data_col_name] = False
        df_shuffled.loc[:int(len(df) * self.test_train_split_pct), test_data_col_name] = True  # TODO: med/high: ensure that the holdout percent is pull

        df_shuffled = df_shuffled.reset_index()
        self.df_features_train_scaled = df_shuffled

        return self
    # Saving and stuff
    def save(self, output_path_dir=None):
        """
        TODO: low: explain
        :param output_path_dir: (str) an absolute path to a DIRECTORY where the pipeline will be saved.
        """
        if output_path_dir is None:
            output_path_dir = config.OUTPUT_PATH if self._source_folder is None else self._source_folder
        logger.debug(f'Attempting to save pipeline to the following folder: {output_path_dir}.')
        # Check if valid directory
        if not os.path.isdir(output_path_dir):
            raise ValueError(f'Invalid output path dir specified: {output_path_dir}')
        final_out_path = os.path.join(output_path_dir, generate_pipeline_filename(self._name))

        # Check if valid final path to be saved
        if not check_arg.is_pathname_valid(final_out_path):
            invalid_path_err = f'Invalid output path save: {final_out_path}'
            logger.error(invalid_path_err)
            raise ValueError(invalid_path_err)

        # Write to file
        with open(final_out_path, 'wb') as model_file:
            joblib.dump(self, model_file)
        # Since successful, save
        self._source_folder = output_path_dir

        logger.debug(f'Pipeline ({self.name}) saved to: {final_out_path}')
        return self
    def has_been_previously_saved(self) -> bool:
        if not self._source_folder: return False
        if not os.path.isfile(os.path.join(self._source_folder, generate_pipeline_filename(self.name))): return False
        return True
    # Video stuff
    def write_video_frames_to_disk(self, video_to_be_labeled=config.VIDEO_TO_LABEL_PATH):
        labels = list(self.df_features_train_scaled[self.gmm_assignment_col_name].values)
        labels = [f'label_i={i} // label={l}' for i, l in enumerate(labels)]
        # util.videoprocessing.write_annotated_frames_to_disk_from_video_LEGACY(
        #     video_to_be_labeled,
        #     labels,
        # )
        # TODO: ensure new implementation works!!!!!
        videoprocessing.write_annotated_frames_to_disk_from_video_NEW(
            video_to_be_labeled,
            labels,
        )
        return
    def make_video(self, video_to_be_labeled=config.VIDEO_TO_LABEL_PATH, output_fps=config.OUTPUT_VIDEO_FPS):
        labels = list(self.df_features_train_scaled[self.gmm_assignment_col_name].values)
        labels = [f'label={l}' for l in labels]
        if not os.path.isfile(video_to_be_labeled):
            not_a_video_err = f'The video to be labeled is not a valid file/path. ' \
                              f'Submitted video path: {video_to_be_labeled}. '
            logger.error(not_a_video_err)
        else:
            self.write_video_frames_to_disk()
            # Write video

            # self.make_video_from_written_frames()

        return self
    # Diagnostics and graphs
    def get_plot_svm_assignments_distribution(self) -> Tuple[object, object]:
        fig, ax = visuals.plot_assignment_distribution_histogram(
            self.df_features_train_scaled[self.svm_assignment_col_name])
        return fig, ax
    def diagnostics(self) -> str:
        """ Function for displaying current state of pipeline. Useful for diagnostics. """
        diag = f"""
self.is_built: {self.is_built}
unique sources in df_train GMM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.gmm_assignment_col_name].values))}
unique sources in df_train SVM ASSIGNMENTS: {len(np.unique(self.df_features_train[self.svm_assignment_col_name].values))}
self._has_unengineered_training_data: {self._has_unengineered_training_data}
len(self._dfs_list_raw_train_data): {len(self._dfs_list_raw_train_data)}
self.train_data_files_paths: {self.train_data_files_paths}
""".strip()
        return diag
    def plot_assignments_in_3d(self, show_now=False, save_to_file=False, azim_elev: Tuple[int, int] = (70, 135)):
        """

        :param show_now:
        :param save_to_file:
        :param azim_elev:
        :return:
        """
        logger.error(f'ERROR: {inspect.stack()[0][3]}(): TODO: IMPLEMENT.')
        # TODO: low: check for other runtiem vars
        if not self.is_built:  # or not self._has_unused_raw_data:
            e = f'Classifiers have not been built. Nothing to graph.'
            logger.warning(e)
            raise ValueError(e)

        fig, ax = visuals.plot_GM_assignments_in_3d_tuple(
            self.df_features_train_scaled[self.dims_cols_names].values,
            self.df_features_train_scaled[self.gmm_assignment_col_name].values,
            save_to_file,
            show_now=show_now,
            azim_elev=azim_elev,
        )
        return fig  # self.fig_gm_assignments_3d
    def plot_cross_val_scoring(self):
        not_implemented = f''
        logger.error(not_implemented)
        # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')
        # util.visuals.plot_accuracy_SVM(scores, fig_file_prefix='TODO__replaceThisFigFilePrefixToSayIfItsKFOLDAccuracyOrTestAccuracy')
        raise NotImplementedError(not_implemented)
    # Legacy stuff. Potential deprecation material.
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


# Concrete pipeline implementations

class PipelinePrime(BasePipeline):
    """
    First implementation of a pipeline. Utilizes the 7 original features from the original B-SOiD paper.

    Use DataFrames instead of unnamed numpy arrays like the previous iteration

    For a list of valid kwarg parameters, check parent object, "BasePipeline".
    """
    def __init__(self, name=None, data_source: str = None, tsne_source: str = 'sklearn', data_ext=None, **kwargs):
        super().__init__(name=name, data_source=data_source, tsne_source=tsne_source, data_ext=data_ext, **kwargs)

    # Higher level data processing functions
    def tsne_reduce_df_features_train(self):
        arr_tsne_result = self.train_tsne_get_dimension_reduced_data(self.df_features_train)
        self.df_features_train_scaled = pd.concat([
            self.df_features_train_scaled,
            pd.DataFrame(arr_tsne_result, columns=self.dims_cols_names),
        ], axis=1)
        return self

    # Pipeline-building functions
    @config.deco__log_entry_exit(logger)
    def build_predict_data(self):
        if not self.is_built:
            self.build_classifier()
        # Engineer predict features
        if self._has_unengineered_predict_data:
            self.engineer_features_predict()
        # Scale

        return self

    # Engineering features
    def engineer_features(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The MAIN function that will build features for BOTH training and prediction data. This
        ensures that processed data for training and prediction are processed in the same way.
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints
        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']

        list_dfs_raw_data: List[pd.DataFrame] = dfs

        # Reconcile args
        if isinstance(list_dfs_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_raw_data, ]
        if not isinstance(list_dfs_raw_data, list):
            raise TypeError(f'Invalid type found: {type(list_dfs_raw_data)} // TODO: elaborate')

        # Adaptively filter features
        dfs_list_adaptively_filtered: List[Tuple[pd.DataFrame, List[float]]] = [
            feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]

        # Engineer features as necessary
        dfs_features: List[pd.DataFrame] = []
        for df_i, _ in tqdm(dfs_list_adaptively_filtered, desc='Engineer features...'):
            # Save scorer, source values because the current way of engineering features strips out that info.
            # scorer, source, file_source, data_source = df_i['scorer'][0], df_i['source'][0], df_i['file_source'][0], df_i['data_source'][0]

            df_features_i = feature_engineering.engineer_7_features_dataframe_NOMISSINGDATA(
                df_i, features_names_7=self.features_names_7)
            for col in columns_to_save:
                if col not in df_features_i.columns and col in df_i.columns:
                    df_features_i[col] = df_i[col].values

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

        # Aggregate all data
        df_features = pd.concat(dfs_features)

        return df_features

    @config.deco__log_entry_exit(logger)
    def engineer_features_train(self) -> BasePipeline:
        """
        Utilizes
        All functions that take the raw data (data retrieved from using bsoid.read_csv()) and
        transforms it into classifier-ready data.

        :return:
        (Includes scorer, source cols)
        """
        # TODO: low: save feature engineering time for train data
        start = time.perf_counter()
        # Queue data
        list_dfs_raw_data: List[pd.DataFrame] = self._dfs_list_raw_train_data
        # Call engineering function
        df_features = self.engineer_features(list_dfs_raw_data)
        # Save data, return
        self.df_features_train = df_features
        end = time.perf_counter()
        # TODO: low: save time it took to engineer train feature-?
        return self

    @config.deco__log_entry_exit(logger)
    def engineer_features_predict(self) -> BasePipeline:
        """ TODO
        """
        # Queue data
        list_dfs_raw_data: List[pd.DataFrame] = self._dfs_list_raw_predict_data
        # Call engineering function
        df_features = self.engineer_features(list_dfs_raw_data)
        # Save data, return
        self.df_features_predict = df_features
        self._has_unengineered_predict_data = False
        return self

    # Build classifier
    @config.deco__log_entry_exit(logger)
    def build_classifier(self, reengineer_train_features: bool = False) -> BasePipeline:
        """
        Builds the model for predicting behaviours.
        TODO: low: elaborate further.
        """
        # Engineer features
        logger.debug(f'{inspect.stack()[0][3]}(): Start engineering features')
        if reengineer_train_features or self._has_unengineered_training_data:
            self.engineer_features_train()

        # Scale data
        self.scale_transform_train_data(create_new_scaler=True)  # Optional kwarg specified for clarity

        # TSNE -- create new dimensionally reduced data
        self.tsne_reduce_df_features_train()

        # Train GMM, get assignments
        self.train_GMM(self.df_features_train_scaled[self.dims_cols_names])

        self.df_features_train_scaled[self.gmm_assignment_col_name] = self.clf_gmm.predict(
            self.df_features_train_scaled[self.dims_cols_names].values)

        # Test-train split
        self.add_test_data_column_to_scaled_train_data()

        # # Train SVM
        self.train_SVM()
        self.df_features_train_scaled[self.svm_assignment_col_name] = self.clf_svm.predict(
            self.df_features_train_scaled[self.features_names_7].values)

        # # Get cross-val accuracy scores
        self.cross_val_scores = cross_val_score(
            self.clf_svm,
            self.df_features_train_scaled[self.features_names_7],
            self.df_features_train_scaled[self.svm_assignment_col_name],
        )
        df_features_train_scaled_test_data = self.df_features_train_scaled.loc[~self.df_features_train_scaled[self.test_col_name]]
        self._acc_score = accuracy_score(
            y_pred=self.clf_svm.predict(df_features_train_scaled_test_data[self.features_names_7]),
            y_true=df_features_train_scaled_test_data[self.svm_assignment_col_name].values)
        logger.debug(f'Pipeline train accuracy: {self.accuracy_score}')

        # Final touches. Save state of pipeline.
        self._is_built = True
        self._has_unengineered_training_data = False
        logger.debug(f'All done with building classifiers!')
        return self

    # Generating predict data
    @config.deco__log_entry_exit(logger)
    def generate_predict_data_assignments(self, reengineer_train_data_features: bool = False, reengineer_predict_features = False) -> BasePipeline:  # TODO: low: rename?
        """ Runs after build(). Using terminology from old implementation. TODO: purpose """
        # TODO: add arg checking for empty predict data?

        # Check that classifiers are built on the training data
        if not self.is_built or reengineer_train_data_features or self._has_unengineered_training_data:
            self.build_classifier()

        # Check if predict features have been engineered
        if reengineer_predict_features or self._has_unengineered_predict_data:
            self.engineer_features_predict()
            self.scale_transform_predict_data()

        # Add prediction labels
        self.df_features_predict_scaled[self.svm_assignment_col_name] = self.clf_svm.predict(
            self.df_features_predict_scaled[self.features_names_7].values)

        return self

    @config.deco__log_entry_exit(logger)
    def build(self, reengineer_train_features=False, reengineer_predict_features=False):
        """
        Build all classifiers and get predictions from predict data
        """
        # Build model
        self.build_classifier(reengineer_features=reengineer_train_features)
        # Get predict data
        self.generate_predict_data_assignments(reengineer_predict_features=reengineer_predict_features)
        return self


# Accessory functions

def generate_pipeline_filename(name):
    """
    Generates a pipeline file name given its name. This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


# Debugging efforts

if __name__ == '__main__':
    # This __main__ section is a debugging effort and holds no value to the final product.
    BSOID = os.path.dirname(os.path.dirname(__file__))
    if BSOID not in sys.path: sys.path.append(BSOID)
    test_file_1 = "C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos\\" \
                  "Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
    test_file_2 = "C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos\\" \
                  "Video2DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
    assert os.path.isfile(test_file_1)
    assert os.path.isfile(test_file_2)

    run = True
    if run:
        # Test build
        nom = 'prime21'
        loc = 'C:\\Users\\killian\\Pictures'
        full_loc = os.path.join(loc, f'{nom}.pipeline')
        actual_save_loc = f'C:\\{nom}.pipeline'

        make_new = True
        save_new = True
        if make_new:
            p = PipelinePrime(name=nom)
            p = p.add_train_data_source(test_file_1)
            p = p.add_predict_data_source(test_file_2)
            p = p.build_classifier(True)
            p = p.generate_predict_data_assignments()
            if save_new:
                p = p.save(loc)
            print(f'cross_val_scores: {p.cross_val_scores}')

        read_existing = False
        if read_existing:
            # p = bsoid.read_pipeline(actual_save_loc)
            p = io.read_pipeline(full_loc)
            print(str(p.train_data_files_paths))
            p.plot_assignments_in_3d(show_now=True)

        pass


# C:\\Users\\killian\\projects\\OST-with-DLC\\bsoid_train_videos

