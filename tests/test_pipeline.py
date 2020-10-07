"""

"""

from typing import List
from unittest import TestCase, skip
import itertools
import numpy as np
import os
import pandas as pd

import bsoid

# test_file_name_for_7_features = 'FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
# test_file_name = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name_for_7_features)
test_file_name = bsoid.config.DEFAULT_TEST_FILE


class TestPipeline(TestCase):

    def test__stub(self):
        """

        """
        # Arrange

        # Act
        is_equal = 1 + 1 == 2
        # Assert
        self.assertTrue(is_equal)



