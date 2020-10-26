"""

"""
from typing import List
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd

import bsoid

test_file_name = bsoid.config.DEFAULT_CSV_TEST_FILE
assert os.path.isfile(test_file_name)


class TestPipeline(TestCase):

    # SCALING DATA
    def test__scale_data__shouldReturnDataFrameWithSameColumnNames__afterScalingData(self):
        """

        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('TestPipeline').add_predict_data_source(test_file_name).build(True)

        # Act
        p = p.scale_transform_train_data()

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

    # ADDING NEW TRAIN DATA SOURCES
    def test__pipeline_adding_train_data_file_source__should____(self):  # TODO: low: add should RE: increment
        """"""
        # Arrange
        data_source_file_path = test_file_name
        p = bsoid.pipeline.PipelinePrime('Test')
        list_of_sources_before_addition: int = len(p._dfs_list_raw_train_data)

        # Act
        p = p.add_train_data_source(data_source_file_path)
        list_of_sources_after_addition: int = len(p._dfs_list_raw_train_data)

        # Assert
        is_file_list_incremented_by_one_after_adding_new_source = \
            list_of_sources_before_addition + 1 == list_of_sources_after_addition

        err_msg = f"""
list_of_sources_before_addition = {list_of_sources_before_addition}
list_of_sources_after_addition = {list_of_sources_after_addition}
"""
        self.assertTrue(isinstance(p, bsoid.pipeline.BasePipeline))
        self.assertTrue(is_file_list_incremented_by_one_after_adding_new_source, err_msg)

    def test__pipeline_adding_train_data_file_source__shouldBeZeroToStart(self):  # TODO: low: add should RE: increment
        """"""
        # Arrange

        # Act
        pipeline_65465465465 = bsoid.pipeline.PipelinePrime('Test_65465465465')

        # Assert
        expected_amount_of_dataframes = 0
        actual_amount_of_dataframes = len(pipeline_65465465465._dfs_list_raw_train_data)
        err_msg = f"""
expected_amount_of_dataframes = {expected_amount_of_dataframes}
actual_amount_of_dataframes = {actual_amount_of_dataframes}
dfs = {[x.to_string() for x in pipeline_65465465465._dfs_list_raw_train_data]}
"""
        self.assertEqual(expected_amount_of_dataframes, actual_amount_of_dataframes, err_msg)

    @skip  # TODO: review if test completely built
    def test__pipeline_adding_train_data_file_source__shouldAddParticularFileTo____when____(self):
        """"""
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test123123123')
        data_source_file_path = test_file_name
        self.assertTrue(data_source_file_path not in p.train_data_files_paths)

        # Act
        p = p.add_train_data_source(data_source_file_path)
        is_path_now_in_list_of_paths = data_source_file_path in p.train_data_files_paths
        # Assert

        err_msg = f"""
p.train_data_files_paths = {p.train_data_files_paths}
""".strip()
        self.assertTrue(is_path_now_in_list_of_paths, err_msg)


    @skip
    def test__pipeline_add_train_data__(self):  # TODO: add should/when
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

