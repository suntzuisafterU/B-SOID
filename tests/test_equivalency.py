"""
Test to see if new superseding functions output the same, correct values as legacy functions
"""

from typing import List
from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid


class test__new_functions_against_legacy(TestCase):

    def test__consistency_of_feature_extraction(self):  # TODO: fill in function name later
        """Create apparatus that compares two feature extraction methods.
        This function uses the SAME feature extraction function -- the only thing tested here is that
        the equality apparatus works. If False, check logic! """
        # Arrange
        # # 1/2: Set up data for function use
        test_file_name = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
        body_parts = bsoid.config.BODYPARTS_PY_LEGACY
        fps = bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(
            os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name))
        data_as_array, _ = bsoid.util.likelihoodprocessing.preprocess_data_and_adaptive_filter(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is = bsoid.classify.bsoid_extract_py
        new_datapreprocess_function = bsoid.classify.bsoid_extract_py

        # Act
        features_output_original_function: List[np.ndarray] = bsoid_py_extract_function__as_is([data_as_array],
                                                                                               body_parts, fps)
        features_output_new_function: List[np.ndarray] = new_datapreprocess_function([data_as_array], body_parts,
                                                                                     fps)

        # Assert (Note: usually multiple asserts in a single test is bad form, but we can refactor this test later)
        # 1/2: Assert types first
        is_old_output_list = isinstance(features_output_original_function, list)
        is_new_output_list = isinstance(features_output_new_function, list)
        self.assertTrue(is_old_output_list)
        self.assertTrue(is_new_output_list)
        # # 2/2: Assert outcomes are equal second

        is_features_data_output_equal = False not in [(a1 == a2).all() for a1, a2 in zip(features_output_original_function, features_output_new_function)]
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {(features_output_original_function[0].shape)}
new output array shape: {(features_output_new_function[0].shape)}

original output array: {features_output_original_function}
new output array: {features_output_new_function}
""".strip()
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    def test__old_vs_new_feature_extraction(self):  # TODO: fill in function name later
        # Arrange
        # # 1/2: Set up data for function use
        test_file_name = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
        body_parts = bsoid.config.BODYPARTS_PY_LEGACY
        fps = bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(
            os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name))
        data_as_array, _ = bsoid.util.likelihoodprocessing.preprocess_data_and_adaptive_filter(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is: callable = bsoid.classify.bsoid_extract_features_without_assuming_100ms_bin_integration
        # bsoid_py_extract_function__as_is: callable = bsoid.classify.bsoid_extract_py
        new_feature_extraction_function: callable = bsoid.train.extract_7_features_bsoid_tsne_py

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
        is_features_data_output_equal = False not in [(a1 == a2).all() for a1, a2 in
                                                      zip(features_output_original_function,
                                                          features_output_new_function)]
        arrays_not_equal_err = f"""
Arrays not identical.
original output array shape: {(features_output_original_function[0].shape)}
new output array shape: {(features_output_new_function[0].shape)}

original output array: {features_output_original_function}
new output array: {features_output_new_function}
""".strip()
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)

    def test__legacy_bsoid_extract_has_same_output_as_functionally_segregated_equivalent(self):  # TODO: fill in function name later
        """Create apparatus that compares two feature extraction methods.
        This function uses the SAME feature extraction function -- the only thing tested here is that
        the equality apparatus works. If False, check logic! """
        # Arrange
        # # 1/2: Set up data for function use
        test_file_name = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
        body_parts = bsoid.config.BODYPARTS_PY_LEGACY
        fps = bsoid.config.VIDEO_FPS

        df_input_data = pd.read_csv(
            os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name))
        data_as_array, _ = bsoid.util.likelihoodprocessing.preprocess_data_and_adaptive_filter(df_input_data)
        # # 2/2: Tee up functions to be compared
        bsoid_py_extract_function__as_is = bsoid.classify.bsoid_extract_py
        first_extraction_seg_func = bsoid.classify.bsoid_extract_features_without_assuming_100ms_bin_integration
        second_extraction_seg_func = bsoid.classify.integrate_features_into_100ms_bins

        # Act
        features_output_original_function: List[np.ndarray] = bsoid_py_extract_function__as_is([data_as_array],
                                                                                               body_parts, fps)
        features_output_new_function: List[np.ndarray] = first_extraction_seg_func([data_as_array], body_parts, fps)
        features_output_new_function = second_extraction_seg_func([data_as_array], features_output_new_function, fps)

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
original output array shape: {(features_output_original_function[0].shape)}
new output array shape: {(features_output_new_function[0].shape)}

original output array: {features_output_original_function}
new output array: {features_output_new_function}
""".strip()
        self.assertTrue(is_features_data_output_equal, msg=arrays_not_equal_err)
    @skip
    def test__sample(self):

        pass


