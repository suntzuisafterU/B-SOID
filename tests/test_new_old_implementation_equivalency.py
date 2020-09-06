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

test_file_name_for_7_features = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
test_file_location_7feat = os.path.join(
    bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name_for_7_features)


class TestNewFunctionEquivalencyToLegacy(TestCase):

    def test__old_vs_new_feature_extraction__bsoid_py(self):
        """

        """
        # Arrange
        # # 1/2: Set up data for function use
        body_parts, fps = bsoid.config.BODYPARTS_PY_LEGACY, bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(test_file_location_7feat, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is: callable = bsoid.classify.bsoid_extract_features_py_without_assuming_100ms_bin_integration
        new_feature_extraction_function: callable = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py

        # Act
        features_output_original_function: List[np.ndarray] = bsoid_py_extract_function__as_is([data_as_array], body_parts, fps)
        features_output_new_function: List[np.ndarray] = new_feature_extraction_function([data_as_array], body_parts, fps)

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



    def test__legacy_bsoid_extract_has_same_output_as_functionally_segregated_equivalent(self):  # TODO: fill in function name later
        """The original implementation for bsoid_extract (_py submodule) assumed that the user wants features
        further filtered to 100ms bins. The original function was separated into 2 new functions and
        this test aims to confirm the correctness that pipelining data from the first and second new functions is
        equivalent to using the original implementation."""
        # Arrange
        # # 1/2: Set up data for function use
        test_file_name = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
        body_parts = bsoid.config.BODYPARTS_PY_LEGACY
        fps = bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(
            os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name),
            nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is = bsoid.classify.bsoid_extract_py
        first_extraction_seg_func = bsoid.classify.bsoid_extract_features_py_without_assuming_100ms_bin_integration
        second_extraction_seg_func = bsoid.classify.integrate_features_into_100ms_bins

        # Act
        features_output_original_function: List[np.ndarray] = bsoid_py_extract_function__as_is([data_as_array], body_parts, fps)

        features_output_new_function_1_of_2_done: List[np.ndarray] = first_extraction_seg_func([data_as_array], body_parts, fps)
        features_output_new_function = second_extraction_seg_func([data_as_array], features_output_new_function_1_of_2_done, fps)

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

    def test__consistency_of_feature_extraction(self):  # TODO: fill in function name later
        """Purpose: create apparatus that compares two feature extraction methods.
        This function uses the SAME feature extraction function -- the only thing tested here is that
        the equality apparatus works. If False, check logic! """
        # Arrange
        # # 1/2: Set up data for function use
        body_parts, fps = bsoid.config.BODYPARTS_PY_LEGACY, bsoid.config.VIDEO_FPS
        df_input_data = pd.read_csv(test_file_name_for_7_features, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is = bsoid.classify.bsoid_extract_py
        new_datapreprocess_function = bsoid.classify.bsoid_extract_py

        # Act
        features_output_original_function: List[np.ndarray] = bsoid_py_extract_function__as_is(
            [data_as_array], body_parts, fps)
        features_output_new_function: List[np.ndarray] = new_datapreprocess_function(
            [data_as_array], body_parts, fps)

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)
        # 1/2: Assert types first
        is_old_output_list = isinstance(features_output_original_function, list)
        is_new_output_list = isinstance(features_output_new_function, list)
        self.assertTrue(is_old_output_list)
        self.assertTrue(is_new_output_list)
        # # 2/2: Assert outcomes are equal second

        is_features_data_output_equal = False not in [(a1 == a2).all() for a1, a2 in
                                                      zip(features_output_original_function,
                                                          features_output_new_function)]
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {features_output_original_function[0].shape}
new output array shape: {features_output_new_function[0].shape}

original output array: {features_output_original_function}
new output array: {features_output_new_function}
""".strip()
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    @skip
    def test__old_vs_new_feature_extraction__bsoid_voc(self):
        """
        TODO: Finish implementation.
            - Needs a proper file to use as test data
            - Needs to have "new" function correctly segregated and added as the function to test against
        """
        # Arrange
        # # 1/2: Tee up functions to be compared
        train_umap_unsupervised_umapapp___as_is: callable = bsoid.train.train_umap_unsupervised_with_xy_features_umapapp
        new_feature_extraction_function: callable = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py

        # # 2/2: Set up data for function use
        body_parts, fps = bsoid.config.BODYPARTS_PY_LEGACY, bsoid.config.VIDEO_FPS
        df_input_data = pd.read_csv(test_file_location_7feat, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        data_as_array, _ = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively(df_input_data)

        # Act
        features_output_original_function: List[np.ndarray] = train_umap_unsupervised_umapapp___as_is(
            [data_as_array, ], fps)
        features_output_new_function: List[np.ndarray] = new_feature_extraction_function(
            [data_as_array, ], fps)

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


