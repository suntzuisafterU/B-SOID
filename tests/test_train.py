from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid


# test_file_name = 'FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
# single_test_file_location = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name)
test_file_name = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH


class TestTrain(TestCase):

    def test__REPLACETHISFUNCTIONNAME(self):
        # Arrange
        df_input_data = pd.read_csv(test_file_name, nrows=bsoid.config.max_rows_to_read_in_from_csv)

        original_datapreprocess_function_ADAPTIVE_FILTER = bsoid.feature_engineering.adaptive_filter_LEGACY
        new_datapreprocess_function = bsoid.feature_engineering.process_raw_data_and_filter_adaptively

        # Act
        original_array_output, _ = original_datapreprocess_function_ADAPTIVE_FILTER(df_input_data)
        new_array_output, _ = new_datapreprocess_function(df_input_data)

        # Assert
        is_filtered_data_equal = np.array_equal(original_array_output, new_array_output)
        self.assertTrue(is_filtered_data_equal)


