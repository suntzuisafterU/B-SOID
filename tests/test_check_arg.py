from typing import List
from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid


########################################################################################################################

# test_file_name = 'TRUNCATED_sample__Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000 - Copy.csv'
# test_file_name = 'FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
# test_file_name = 'RowsDeleted_FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv'
test_file_name = bsoid.config.DEFAULT_CSV_TEST_FILE

single_test_file_location = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name)


class TestCheckArg(TestCase):

    def test__(self):
        # Arrange
        self.assertRaises()

        # Act

        # Assert

        self.assertEqual(None, None)
        return





