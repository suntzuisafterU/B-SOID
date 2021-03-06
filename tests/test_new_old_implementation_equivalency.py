"""
Test to see if new superseding functions output the same, correct values as legacy functions
"""

from typing import List
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd

import bsoid

# test_file_name_for_7_features = 'FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
# test_file_name = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name_for_7_features)
test_file_name = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH


class TestNewFunctionEquivalencyToLegacy(TestCase):

    def test__old_vs_new_feature_extraction__bsoid_py(self):  # TODO: not passing. Delete this when fixed.
        """
        # TODO
        """
        # Arrange
        # # 1/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is: callable = \
            bsoid.classify.bsoid_extract_features_py_without_assuming_100ms_bin_integration
        new_feature_extraction_function: callable = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py
        # # 2/2: Set up data for function use
        df_input_data = pd.read_csv(test_file_name, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        array_data, _ = bsoid.feature_engineering.process_raw_data_and_filter_adaptively(df_input_data)

        # Act
        features_output_original: np.ndarray = bsoid_py_extract_function__as_is([array_data, ])[0].T
        features_output_new: np.ndarray = new_feature_extraction_function([array_data, ])[0].T

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)
        # # 1/1: Assert outcomes are equal second
        is_features_data_output_equal = (features_output_original == features_output_new).all()
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {features_output_original.shape}
new output array shape: {features_output_new.shape}

---

original output array: {features_output_original}
new output array: {features_output_new}

---

diff:
{features_output_new - features_output_original}


"""
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    def test__legacy_bsoid_extract_has_same_output_as_functionally_segregated_equivalent(self):  # TODO: not passing. Delete this when fixed.
        """
        The original implementation for bsoid_extract (_py submodule) assumed that the user wants features
        further filtered to 100ms bins. In response, the original function was separated into 2 new functions and
        this test aims to confirm the correctness that pipelining data from the first and second new functions is
        equivalent to using the original implementation.
        """
        # Arrange
        # # 1/2: Tee up functions to be compared
        # Old way
        bsoid_py_extract_function__original = bsoid.classify.bsoid_extract_py
        # New way
        first_extraction_seg_func = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py
        second_extraction_seg_func = bsoid.feature_engineering.integrate_features_into_100ms_bins_LEGACY

        # # 2/2: Set up data for function use
        body_parts = bsoid.config.BODYPARTS_PY_LEGACY
        fps = bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(test_file_name, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.feature_engineering.process_raw_data_and_filter_adaptively(df_input_data)

        # Act
        # # Get outcome of original function
        features_output_original: np.ndarray = bsoid_py_extract_function__original(
            [data_as_array, ], body_parts, fps)[0]
        # # Get outcomes of new segregated functions
        features_output_new_function_1_of_2_done: List[np.ndarray] = first_extraction_seg_func(
            [data_as_array, ], body_parts, fps)
        features_output_new: np.ndarray = second_extraction_seg_func(
            [data_as_array, ], features_output_new_function_1_of_2_done, fps)[0]

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)

        # # 1/1: Assert outcomes are equal second
        is_features_data_output_equal = (features_output_original == features_output_new).all()
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {features_output_original.shape}
new output array shape: {features_output_new.shape}

original output array: {features_output_original}
new output array: {features_output_new}

original output array: {features_output_original}
new output array: {features_output_new}

diff (new - old):
{features_output_new - features_output_original}
"""
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    def test__consistency_of_feature_extraction(self):  # TODO: fill in function name later
        """

        """
        # Arrange
        # # 1/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is = bsoid.classify.bsoid_extract_py
        new_data_preprocess_function = bsoid.classify.bsoid_extract_py

        # # 2/2: Set up data for function use
        body_parts, fps = bsoid.config.BODYPARTS_PY_LEGACY, bsoid.config.VIDEO_FPS
        df_input_data = pd.read_csv(test_file_name, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.feature_engineering.process_raw_data_and_filter_adaptively(df_input_data)

        # Act
        original_function_features_output: List[np.ndarray] = bsoid_py_extract_function__as_is(
            [data_as_array, ], body_parts, fps)
        new_function_features_output: List[np.ndarray] = new_data_preprocess_function(
            [data_as_array, ], body_parts, fps)

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)
        # 1/2: Assert types first
        is_old_output_list = isinstance(original_function_features_output, list)
        self.assertTrue(is_old_output_list)
        is_new_output_list = isinstance(new_function_features_output, list)
        self.assertTrue(is_new_output_list)

        # # 2/2: Assert outcomes are equal second
        is_features_data_output_equal = False not in [
            (a1 == a2).all() for a1, a2 in zip(original_function_features_output, new_function_features_output)]
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {original_function_features_output[0].shape}
new output array shape: {new_function_features_output[0].shape}

original output array: {original_function_features_output}
new output array: {new_function_features_output}
"""
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    @skip
    def test__old_vs_new_feature_extraction__bsoid_voc(self):
        """
        TODO: NOT YET CORRECTLY IMPLEMENTED. Finish test implementation.
            - Needs a proper file to use as test data
            - Needs to have "new" function correctly segregated and added as the function to test against
        """
        # Arrange
        # # 1/2: Tee up functions to be compared
        train_umap_unsupervised_umapapp___as_is: callable = bsoid.train_LEGACY.train_umap_unsupervised_with_xy_features_umapapp
        new_feature_extraction_function: callable = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py

        # # 2/2: Set up data for function use
        body_parts, fps = bsoid.config.BODYPARTS_PY_LEGACY, bsoid.config.VIDEO_FPS
        df_input_data = pd.read_csv(test_file_name, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively(df_input_data)

        # Act
        features_output_original_function: List[np.ndarray] = train_umap_unsupervised_umapapp___as_is(
            [data_as_array, ], fps)
        features_output_new_function: List[np.ndarray] = new_feature_extraction_function(
            [data_as_array, ], fps=fps)

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)
        # 1/2: Assert types first
        is_old_output_list = isinstance(features_output_original_function, list)
        is_new_output_list = isinstance(features_output_new_function, list)
        self.assertTrue(is_old_output_list)
        self.assertTrue(is_new_output_list)

        # # 2/2: Assert outcomes are equal second
        is_features_data_output_equal = False not in [
            (a1 == a2).all() for a1, a2 in zip(features_output_original_function, features_output_new_function)]
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {features_output_original_function[0].shape}
new output array shape: {features_output_new_function[0].shape}

original output array: {features_output_original_function}
new output array: {features_output_new_function}
""".strip()
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    @skip
    def test__sample(self):

        pass


