"""

"""
from typing import Any, List
from unittest import TestCase, skip
import numpy as np
import os
import pandas as pd

import bsoid

csv_test_file_path = bsoid.config.DEFAULT_CSV_TEST_FILE
h5_test_file_path = bsoid.config.DEFAULT_H5_TEST_FILE
assert os.path.isfile(csv_test_file_path)
assert os.path.isfile(h5_test_file_path)


class TestIO(TestCase):

    # Reading data
    @skip  # TODO: ensure that bsoid.io.read_h5 is implemented before removing this @skip annotation
    def test__read_h5_read_csv__shouldReturnTheSameOutput__whenBaseDataIsSameButExtensionIsTheOnlyDifference(self):
        """

        """
        # Arrange
        csv_data_source_file_path = csv_test_file_path
        h5_data_source_file_path = h5_test_file_path

        # Act
        read_csv_output: np.ndarray = bsoid.io.read_csv(csv_data_source_file_path).values
        read_h5_output: np.ndarray = bsoid.io.read_h5(h5_data_source_file_path).values

        are_numpy_arrays_equal = (read_csv_output == read_h5_output).all()
        # Assert
        err_message = f"""
read_csv_output = 
{read_csv_output}

read_h5_output = 
{read_h5_output}

diff:
{read_csv_output - read_h5_output}
""".strip()
        self.assertTrue(are_numpy_arrays_equal, err_message)