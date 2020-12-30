from typing import List
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd

import bsoid


########################################################################################################################

# test_file_name = 'TRUNCATED_sample__Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000 - Copy.csv'
# test_file_name = 'FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
# test_file_name = 'RowsDeleted_FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
test_file_name = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH

single_test_file_location = test_file_name  # os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name)
assert os.path.isfile(single_test_file_location), f'Could not find test file: {single_test_file_location}'


class TestFeatureEngineering(TestCase):

    @skip  # TODO: fix this test; currently unfinished
    def test__adaptively_filter_dlc_output__(self):
        # TODO: ensuring that cols like scorer, file_source, etc don't go missing
        # Arrange
        df_input = bsoid.io.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)

        # Act

        # Assert

        self.assertEqual(None, None)
        return

    def test__adaptively_filter_dlc_output__shouldReturnSameNumberOfRowsAsInput__always(self):
        # Arrange
        df_input = bsoid.io.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        input_num_rows = len(df_input)
        # Act
        df_output, _ = bsoid.feature_engineering.adaptively_filter_dlc_output(df_input)
        output_num_rows = len(df_output)

        # Assert
        err_msg = f"""
{df_input.to_string()}

{df_output.to_string()}
TODO: improve error message
""".strip()
        self.assertEqual(input_num_rows, output_num_rows, err_msg)

    @skip  # TODO: med: address why this fails due to operand shape problems
    def test___newFEGoodAsOldFE(self):
        # Arrange
        # Set up functions for feature engineering
        old_feature_engineer: callable = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py
        new_feature_engineer: callable = bsoid.feature_engineering.engineer_7_features_dataframe_MISSING_1_ROW
        # Read in data
        df_input_data_original = pd.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        arr_input_data_original_filtered, _ = bsoid.feature_engineering.process_raw_data_and_filter_adaptively(df_input_data_original)
        # input_data_original_ready: List[np.ndarray] = bsoid.feature_engineering.extract_7_features_bsoid_tsne_py(
        #     [arr_input_data_original_filtered, ])

        df_input_data_new = bsoid.io.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        df_input_data_new_filtered, _ = bsoid.feature_engineering.adaptively_filter_dlc_output(df_input_data_new)

        # Act
        old_result_list_arrays: List[np.ndarray] = old_feature_engineer([arr_input_data_original_filtered, ])
        old_result_final: np.ndarray = old_result_list_arrays[0].T

        new_result: pd.DataFrame = new_feature_engineer(df_input_data_new_filtered)
        new_result_final: np.ndarray = new_result.values

        # Assert
        arrays_are_equal: bool = np.array_equal(old_result_final, new_result_final)
        top_n_rows, bottom_n_rows = 3, 3
        fail_msg = f"""
Old array shape:
{old_result_final.shape}

New array shape:
{new_result_final.shape}

---

Old result top {top_n_rows} rows:
{old_result_final[:top_n_rows, :]}

New result top {top_n_rows} rows:
{new_result_final[:top_n_rows, :]}

---

Old result bottom {bottom_n_rows} rows:
{old_result_final[:bottom_n_rows, :]}

New result bottom {bottom_n_rows} rows:
{new_result_final[:bottom_n_rows, :]}

---
DIFF:
{new_result_final - old_result_final}
"""
        self.assertTrue(arrays_are_equal, fail_msg)

    @skip  # TODO: med: address why this fails due to operand shape problems
    def test__adaptive_filtering__ensure_new_implementation_using_df_is_correct_compared_to_implementation(self):
        # Arrange
        # Read in data
        df_input_data_original = pd.read_csv(single_test_file_location,
                                             nrows=bsoid.config.max_rows_to_read_in_from_csv)
        df_input_data_new = bsoid.io.read_csv(single_test_file_location,
                                              nrows=bsoid.config.max_rows_to_read_in_from_csv)

        # prep functions
        original_adaptive_filter: callable = bsoid.feature_engineering.process_raw_data_and_filter_adaptively
        new_adaptive_filter: callable = bsoid.feature_engineering.adaptively_filter_dlc_output

        # Act
        arr_original_adaptive_filter_output, _ = original_adaptive_filter(df_input_data_original)
        df_new_adaptive_filter_output, _ = new_adaptive_filter(df_input_data_new)
        df_new_adaptive_filter_output = df_new_adaptive_filter_output.drop('scorer', axis=1)
        arr_new_adaptive_filter_output = np.array(df_new_adaptive_filter_output)

        # Assert
        is_filtered_data_equal = np.array_equal(arr_original_adaptive_filter_output, arr_new_adaptive_filter_output)
        top_n_rows_display, bottom_n_rows_display = 6, 4
        err_not_equal = f"""
------------------------------------------------------------------------------------------------------------------------
Original output shape:
{arr_original_adaptive_filter_output.shape}
New output shape: 
{arr_new_adaptive_filter_output.shape}

---

First {top_n_rows_display} rows of original:
{arr_original_adaptive_filter_output[:top_n_rows_display, :]}

First {top_n_rows_display} rows of new:
{arr_new_adaptive_filter_output[:top_n_rows_display, :]}

---

Last {bottom_n_rows_display} rows of original:
{arr_original_adaptive_filter_output[-bottom_n_rows_display:, :]}

Last {bottom_n_rows_display} rows of new:
{arr_new_adaptive_filter_output[-bottom_n_rows_display:, :]}

---


New raw Data:
{df_input_data_new}

---

Original output:
{arr_original_adaptive_filter_output}

New output:
{arr_new_adaptive_filter_output}

---

Difference:
{arr_original_adaptive_filter_output - arr_new_adaptive_filter_output}
"""

        self.assertTrue(is_filtered_data_equal, err_not_equal)

    def test__win_len(self):  # TODO: high

        pass

    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenOneArraySubmitted(self):
        # TODO:
        # Arrange
        data = [[1, 2, 3],
                [4, 5, 6], ]
        arr_input = np.array(data)
        arr_expected_output = np.array(data)
        # Act
        arr_actual_output = bsoid.feature_engineering.average_vector_between_n_vectors(arr_input)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}
""".strip()  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenTwoArraysSubmitted(self):
        # TODO: finish off this test, then remove this TODO if passes.
        # Arrange
        data_1 = [[0, 2],
                  [2, 2], ]
        data_2 = [[5, 2],
                  [1, 1], ]
        data_expected_output = [[2.5, 2.0],
                                [1.5, 1.5], ]
        arr_input_1 = np.array(data_1)
        arr_input_2 = np.array(data_2)
        arr_expected_output = np.array(data_expected_output)
        # Act
        arr_actual_output = bsoid.feature_engineering.average_vector_between_n_vectors(arr_input_1, arr_input_2)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}

"""  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    @skip  # Temporarily skipped while the test is being finished. It's not finished!
    def test__average_distance_between_n_features__shouldCalculateAverageLocationOfFeature__whenThreeArraysSubmitted(self):
        # TODO: finish the 3rd data set and also the expected data output
        # Arrange
        data_1 = [[0, 2],
                  [2, 2], ]
        data_2 = [[5, 2],
                  [1, 1], ]
        data_3 = [[],  # TODO 1/2
                  [], ]
        data_expected_output = [[2.5, 2.],
                                [1.5, 1.5],
                                [], ]  # TODO 2/2
        arr_input_1 = np.array(data_1)
        arr_input_2 = np.array(data_2)
        arr_input_3 = np.array(data_3)
        arr_expected_output = np.array(data_expected_output)
        # Act
        arr_actual_output = bsoid.feature_engineering.average_vector_between_n_vectors(arr_input_1, arr_input_2, arr_input_3)
        # Assert
        is_equal = (arr_expected_output == arr_actual_output).all()
        err = f"""
Expected output:
{arr_expected_output}

Actual output:
{arr_actual_output}
""".strip()  # TODO: elaborate error message to suss out potential problems
        self.assertTrue(is_equal, err)

    def test__distance_between_2_arrays(self):
        # Arrange
        data_1 = [[5, 2, 3], ]
        arr_1 = np.array(data_1)
        data_2 = [[20, 15.5, 7], ]
        arr_2 = np.array(data_2)
        expected_output_distance: float = 20.573040611440984

        # Act
        actual_output_distance: float = bsoid.feature_engineering.distance_between_two_arrays(arr_1, arr_2)

        # Assert
        err_msg = f"""
expected output: {expected_output_distance}

actual output: {actual_output_distance}
""".strip()
        self.assertEquals(expected_output_distance, actual_output_distance, err_msg)

    def test__average_arr_location(self):
        # Arrange
        data_1 = [[5., 2., 3.], ]
        arr_1 = np.array(data_1)
        data_2 = [[20., 15.5, 7.], ]
        arr_2 = np.array(data_2)
        exp_data = [[(5+20)/2., (2+15.5)/2, (3+7)/2], ]
        expected_output_arr = np.array(exp_data)

        # Act
        actual_output_arr = bsoid.feature_engineering.average_vector_between_n_vectors(arr_1, arr_2)

        # Assert
        is_equals = (expected_output_arr == actual_output_arr).all()
        err_msg = f"""
expected_output_arr output: {expected_output_arr}

actual actual_output_arr: {actual_output_arr}
""".strip()
        self.assertTrue(is_equals, err_msg)

    def test__attach_average_feature_xy__shouldOnlyAttach2ColumnsInResult(self):
        """ Test that the number of output columns matches expected """
        # Arrange
        df = bsoid.io.read_csv(single_test_file_location)
        df_cols_set = set(df.columns)
        output_feature_name = 'AvgFeature'
        expected_num_cols: int = len(df_cols_set) + 2

        # (Ensure col names exist for test)
        feature1, feature2 = 'Forepaw/Shoulder1', 'Forepaw/Shoulder2'
        for feat, xy in itertools.product((feature1, feature2), ('x', 'y')):
            feat_xy = f'{feat}_{xy}'
            assert feat_xy in df_cols_set, f'Column "{feat_xy}" not found in DataFrame. Cannot complete test. Columns = {list(df.columns)}'

        # Act
        df_output: pd.DataFrame = bsoid.feature_engineering.attach_average_bodypart_xy(df, feature1, feature2, output_feature_name, copy=True)
        actual_num_output_cols: int = len(df_output.columns)

        # Assert
        self.assertEqual(expected_num_cols, actual_num_output_cols)



