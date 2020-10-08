"""

"""
from typing import List
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd

import bsoid

test_file_name = bsoid.config.DEFAULT_TEST_FILE
assert os.path.isfile(test_file_name)


class TestPipeline(TestCase):

    def test__scale_data__shouldReturnDataFrameWithSameColumnNames__afterScalingData(self):
        """

        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('TestPipeline').add_predict_data_source(test_file_name).build()

        # Act
        p = p.scale_data()

        unscaled_features_cols = set(p.df_features_train.columns)
        scaled_features_cols = set(p.df_features_train_scaled.columns)
        is_equal = unscaled_features_cols == scaled_features_cols

        # Assert
        err_message = f"""
Cols were found in one but not the other.

unscaled_features_cols = {unscaled_features_cols}
scaled_features_cols = {scaled_features_cols}

Symmetric diff = {unscaled_features_cols.symmetric_difference(scaled_features_cols)}

""".strip()
        self.assertTrue(is_equal, err_message)

    def test__pipeline_adding_file_source__(self):  # TODO: low: add should RE: increment
        """"""
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test')
        data_source_file_path = test_file_name
        list_of_sources_before_addition: int = len(p._dfs_list_raw_train_data)

        # Act
        p = p.add_train_data_source(data_source_file_path)
        list_of_sources_after_addition: int = len(p._dfs_list_raw_train_data)

        # Assert
        is_file_list_incremented_by_one_after_adding_new_source = \
            list_of_sources_before_addition + 1 == list_of_sources_after_addition

        err_msg = f"""

""".strip()
        self.assertTrue(is_file_list_incremented_by_one_after_adding_new_source, err_msg)

    @skip
    def test__stub1(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)
    @skip
    def test__stub2(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)
    @skip
    def test__stub3(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

