from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid


test_file_name = 'TRUNCATED_sample__Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000 - Copy.csv'

single_test_file_location = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name)


class TestFeatureEngineering(TestCase):

    def test__adaptive_filtering__ensure_new_implementation_using_df_is_correct_compared_to_implementation(self):
        # Arrange
        # Read in data
        df_input_data_original = pd.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)
        df_input_data_new = bsoid.io.read_csv(single_test_file_location, nrows=bsoid.config.max_rows_to_read_in_from_csv)

        # prep functions
        original_adaptive_filter: callable = bsoid.util.likelihoodprocessing.process_raw_data_and_filter_adaptively
        new_adaptive_filter: callable = bsoid.feature_engineering.adaptively_filter_dlc_output

        # Act
        arr_original_adaptive_filter_output, _ = original_adaptive_filter(df_input_data_original)
        df_new_adaptive_filter_output, _ = new_adaptive_filter(df_input_data_new)
        df_new_adaptive_filter_output = df_new_adaptive_filter_output.drop('scorer', axis=1)
        arr_new_adaptive_filter_output = np.array(df_new_adaptive_filter_output)

        # Assert
        is_filtered_data_equal = np.array_equal(arr_original_adaptive_filter_output, arr_new_adaptive_filter_output)
        err_not_equal = f"""
------------------------------------------------------------------------------------------------------------------------
Original output shape:
{arr_original_adaptive_filter_output.shape}
New output shape: 
{arr_new_adaptive_filter_output.shape}

---

First two rows of original:
{arr_original_adaptive_filter_output[:2, :]}

First two rows of new:
{arr_new_adaptive_filter_output[:2, :]}

---

Last two rows of original:
{arr_original_adaptive_filter_output[-2:, :]}

Last two rows of new:
{arr_new_adaptive_filter_output[-2:, :]}

---

Original output: {arr_original_adaptive_filter_output}
New output: {arr_new_adaptive_filter_output}
"""

        self.assertTrue(is_filtered_data_equal, err_not_equal)


