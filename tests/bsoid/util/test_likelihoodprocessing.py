from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid


class test__likelihoodprocessing_functions(TestCase):

    def test__adaptive_filtering__(self):
        # Arrange
        test_file_name = 'Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
        df_input_data = pd.read_csv(os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name))

        original_adaptive_filter_func = bsoid.util.likelihoodprocessing.adaptive_filter_data
        new_and_improved_adap_filter_func = bsoid.util.likelihoodprocessing.adaptive_filter_data_new_and_improved

        # Act
        original_array_output, _ = original_adaptive_filter_func(df_input_data)
        new_array_output, _ = new_and_improved_adap_filter_func(df_input_data)

        # Assert
        is_filtered_data_equal = np.array_equal(original_array_output, new_array_output)
        self.assertTrue(is_filtered_data_equal)


