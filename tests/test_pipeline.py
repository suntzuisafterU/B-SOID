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
        p = bsoid.pipeline.PipelinePrime('TestPipeline').add_predict_data_source(test_file_name).engineer_features()

        # Act
        p = p.scale_data()

        unscaled_features_cols = set(p.df_features.columns)
        scaled_features_cols = set(p.df_features_scaled.columns)
        is_equal = unscaled_features_cols == scaled_features_cols

        # Assert
        err_message = f"""
Cols were found in one but not the other.

unscaled_features_cols = {unscaled_features_cols}
scaled_features_cols = {scaled_features_cols}

Symmetric diff = {unscaled_features_cols.symmetric_difference(scaled_features_cols)}

""".strip()
        self.assertTrue(is_equal, err_message)

    @skip
    def test__stub(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)

