"""

"""

from unittest import TestCase, skip
import os

from typing import Any, Dict, List
import itertools
import numpy as np
import pandas as pd

import bsoid

long_csv_test_file_path = "FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
csv_test_file_path = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', long_csv_test_file_path)
assert os.path.isfile(csv_test_file_path)


class TestPipeline(TestCase):

    # SCALING DATA
    def test__scale_data__shouldReturnDataFrameWithSameColumnNames__afterScalingData(self):
        """

        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('TestPipeline').add_predict_data_source(csv_test_file_path).build(True)

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
        data_source_file_path = csv_test_file_path
        p = bsoid.pipeline.PipelinePrime('Test')
        list_of_sources_before_addition: int = len(p._dfs_list_raw_train_data)

        # Act
        p = p.add_train_data_source(data_source_file_path)
        list_of_sources_after_addition: int = len(p._dfs_list_raw_train_data)

        # Assert

        err_msg = f"""
list_of_sources_before_addition = {list_of_sources_before_addition}
list_of_sources_after_addition = {list_of_sources_after_addition}
"""
        self.assertEqual(list_of_sources_before_addition + 1, list_of_sources_after_addition, err_msg)

    def test__pipeline_adding_train_data_file_source__shouldBeZeroToStart(self):  # TODO: low: add should RE: increment
        """"""
        # Arrange
        p1 = bsoid.pipeline.PipelinePrime('Test_65465465465')
        expected_amount_of_dataframes = 0

        # Act
        actual_amount_of_dataframes = len(p1._dfs_list_raw_train_data)

        # Assert
        err_msg = f"""
expected_amount_of_dataframes = {expected_amount_of_dataframes}
actual_amount_of_dataframes = {actual_amount_of_dataframes}

"""  # dfs = {[x.to_string() for x in pipeline_65465465465._dfs_list_raw_train_data]}
        self.assertEqual(expected_amount_of_dataframes, actual_amount_of_dataframes, err_msg)

    @skip  # TODO: review if test completely built
    def test__pipeline_adding_train_data_file_source__shouldAddParticularFileTo____when____(self):
        """"""
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test123123123')
        data_source_file_path = csv_test_file_path
        self.assertTrue(data_source_file_path not in p.train_data_files_paths)

        # Act
        p = p.add_train_data_source(data_source_file_path)
        is_path_now_in_list_of_paths = data_source_file_path in p.train_data_files_paths
        # Assert

        err_msg = f"""
p.train_data_files_paths = {p.train_data_files_paths}
""".strip()
        self.assertTrue(is_path_now_in_list_of_paths, err_msg)

    def test__pipeline_add_train_data__(self):  # TODO: add should/when
        """

        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test_65465465465asdfasdfde34asdf')
        num_sources_before_adding_any = len(p.train_data_files_paths)

        # Act
        p = p.add_train_data_source(csv_test_file_path)
        num_sources_after_adding_sources = len(p.train_data_files_paths)

        is_equal = num_sources_before_adding_any + 1 == num_sources_after_adding_sources
        # Assert
        err_msg = f"""

"""
        self.assertTrue(is_equal, err_msg)

    @skip
    def test__add_train_data_AND_build__shouldHaveSameNumRowsInRawDataAsBuiltData__whenRawDataBuilt(self):
        """
        After adding just 1 train data source,

        *** NOTE: This test usually takes a while since it builds the entire model as part of the test ***

        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime()
        p = p.add_train_data_source(csv_test_file_path)
        original_number_of_data_rows = len(bsoid.read_csv(csv_test_file_path))

        # Act
        p = p.build_classifier()
        actual_total_rows_after_feature_engineering = len(p.df_features_train)

        # Assert
        err_msg = f'TODO: err msg'
        self.assertEqual(original_number_of_data_rows, actual_total_rows_after_feature_engineering, err_msg)

    @skip
    def test__stub3(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

