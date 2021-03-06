"""

"""
from typing import Any, Dict, List, Set
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd
import random
import time

import bsoid


def get_random():
    return random.randint(0, 100_000_000_000)


csv_test_file_path = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH
assert os.path.isfile(csv_test_file_path)


class TestPipeline(TestCase):

    # Param adds, changes, checks
    def test__set_params__shouldKeepDefaultsWhileChangingSpecifiedVars__whenOptionalArgForReadingInConfigVarsNotTrue(self):
        # Arrange
        p = bsoid.pipeline.PipelinePrime('TestPipe07dfdp')
        default_gmm_n_components = p.gmm_n_components

        # Act
        p = p.set_params(tsne_n_components=5)
        gmm_n_components_after_set_param = p.gmm_n_components

        # Assert
        err_msg = f"""

"""
        self.assertEqual(default_gmm_n_components, gmm_n_components_after_set_param, err_msg)

    def test__set_params__shouldDetectVars__whenNewPipelineInstantiatedAsSuch(self):
        # Arrange
        cv = expected_cv = 5

        # Act
        p = bsoid.pipeline.PipelinePrime(f'TestPipeline_{get_random()}', cross_validation_k=cv)
        actual_cv = p.cross_validation_k

        # Assert
        err = f"""Error: cv cross val did not get read-in correctly. TODO: elaborate. """.strip()
        self.assertEqual(expected_cv, actual_cv, err)

    # SCALING DATA
    @skip  # Temporary skip since it takes forever to run this due to sample size
    def test__scale_data__shouldReturnDataFrameWithSameColumnNames__afterScalingData(self):
        # Arrange
        p = bsoid.pipeline.PipelinePrime('TestPipe987dfdp').add_train_data_source(csv_test_file_path).build(True)

        # Act
        p = p.scale_transform_train_data()

        unscaled_features_cols: Set[str] = set(p.df_features_train.columns)
        scaled_features_cols: Set[str] = set(p.df_features_train_scaled.columns)

        # Assert
        err_message = f"""
Cols were found in one but not the other.

unscaled_features_cols = {unscaled_features_cols}
scaled_features_cols = {scaled_features_cols}

Symmetric diff = {unscaled_features_cols.symmetric_difference(scaled_features_cols)}

""".strip()
        self.assertEqual(unscaled_features_cols, scaled_features_cols, err_message)

    # Adding new training data sources
    def test__pipeline_adding_train_data_file_source__should____(self):
        """"""
        # Arrange
        data_source_file_path = csv_test_file_path
        pipe = bsoid.pipeline.PipelinePrime('TestPipeline_asdf')
        num_of_sources_before_addition: int = len(pipe.training_data_sources)
        num_of_sources_should_be_this_after_addition = num_of_sources_before_addition + 1

        # Act
        p = pipe.add_train_data_source(data_source_file_path)
        num_of_sources_actually_this_after_addition: int = len(p.training_data_sources)

        # Assert

        err_msg = f"""
list_of_sources_before_addition = {num_of_sources_before_addition}
num_of_sources_should_be_this_after_addition = {num_of_sources_should_be_this_after_addition}

list_of_sources_after_addition = {num_of_sources_actually_this_after_addition}
"""
        self.assertEqual(num_of_sources_should_be_this_after_addition, num_of_sources_actually_this_after_addition,
                         err_msg)

    def test__pipeline_adding_train_data_file_source__shouldBeZeroToStart(self):
        """"""
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test_6546df5465465')
        expected_amount_of_sources = 0

        # Act
        actual_amount_of_dataframes = len(p.training_data_sources)

        # Assert
        err_msg = f"""
expected_amount_of_dataframes = {expected_amount_of_sources}

actual_amount_of_dataframes = {actual_amount_of_dataframes}
"""
        self.assertEqual(expected_amount_of_sources, actual_amount_of_dataframes, err_msg)

    @skip  # TODO: review if test completely built
    def test__pipeline_adding_train_data_file_source__shouldAddParticularFileTo____when____(self):
        """"""
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test1231asdf23123')
        data_source_file_path = csv_test_file_path

        # Act
        p = p.add_train_data_source(data_source_file_path)
        is_path_now_in_list_of_paths = data_source_file_path in p.train_data_files_paths
        # Assert

        err_msg = f"""
p.train_data_files_paths = {p.train_data_files_paths}
""".strip()
        self.assertTrue(is_path_now_in_list_of_paths, err_msg)

    def test__pipeline_add_train_data__(self):  # TODO: add should/when
        # Arrange
        p = bsoid.pipeline.PipelinePrime('Test_65465465465asddsfasdfde34asdf')
        num_sources_before_adding_any = len(p.training_data_sources)

        # Act
        p = p.add_train_data_source(csv_test_file_path)
        num_sources_after_adding_sources = len(p.training_data_sources)

        is_equal = num_sources_before_adding_any + 1 == num_sources_after_adding_sources
        # Assert
        err_msg = f"""

"""
        self.assertTrue(is_equal, err_msg)

    # @skip
    def test__add_train_data_AND_build__shouldHaveSameNumRowsInRawDataAsBuiltData__whenRawDataBuilt(self):
        """
        After adding just 1 train data source,

        *** NOTE: This test usually takes a while since it builds the entire model as part of the test ***
        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('asdfasdfdfs44444')
        p = p.add_train_data_source(csv_test_file_path)
        original_number_of_data_rows = len(bsoid.io.read_csv(csv_test_file_path))

        # Act
        p = p.build()
        actual_total_rows_after_feature_engineering = len(p.df_features_train)

        # Assert
        err_msg = f'TODO: err msg'
        self.assertEqual(original_number_of_data_rows, actual_total_rows_after_feature_engineering, err_msg)




    def test__get_assignment_label__shouldReturnEmptyString__whenLabelNotSet(self):
        """
        Test to see if output is None if no assignment label found
        """
        # Arrange
        p = bsoid.pipeline.PipelinePrime('APipelineName_asdffdfsdf123987')
        expected_label = ''
        # Act
        actual_label = p.get_assignment_label(0)
        # Assert
        self.assertEqual(expected_label, actual_label)

    def test__set_label__shouldUpdateAssignment__whenUsed(self):
        # Arrange
        p = bsoid.pipeline.PipelinePrime('DeleteMe___6APipelineName12398asdfasdfaasdfdf989dsdf7')
        assignment, input_label = 1, 'Behaviour1'

        # Act
        p = p.set_label(assignment, input_label)

        actual_label = p.get_assignment_label(assignment)
        # Assert
        self.assertEqual(input_label, actual_label)

    def test__updatingAssignment__shouldSaveLabel__whenSavedAndRereadIn(self):
        # Arrange
        name = 'DELETE_ME__APipelineName12398asdzzz1614154fasdfsdf7_'
        p_write = bsoid.pipeline.PipelinePrime(name)
        assignment, input_label = 12, 'Behaviour12'

        # Act
        p_write = p_write.set_label(assignment, input_label)
        p_write.save()

        p_read = bsoid.read_pipeline(os.path.join(
            bsoid.config.OUTPUT_PATH,
            bsoid.pipeline.generate_pipeline_filename(name),
        ))

        actual_label = p_read.get_assignment_label(assignment)
        # Assert
        err = f"""
Expected label: {input_label}

Actual label: {actual_label}


"""  # All labels map: {p_read._map_assignment_to_behaviour}
        self.assertEqual(input_label, actual_label, err)

    @skip
    def test__stub7(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

    @skip
    def test__stub8(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

