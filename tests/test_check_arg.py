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
test_file_name = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH

single_test_file_location = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', test_file_name)


class TestCheckArg(TestCase):

    # ensure_numpy_arrays_are_same_shape
    def test__ensure_numpy_arrays_are_same_shape__ShouldErrorOut__whenArraysDifferentShapes(self):
        # Arrange
        data1 = [[1, 2, 3], [1, 2, 3]]
        data2 = [[1, 2, 3, 4], [1, 2, 3, 4]]
        expected_error = ValueError

        arr1 = np.array(data1)
        arr2 = np.array(data2)
        func = bsoid.check_arg.ensure_numpy_arrays_are_same_shape

        # Act/Assert
        self.assertRaises(expected_error, func, arr1, arr2)

    def test__ensure_numpy_arrays_are_same_shape__shouldRunWithoutError__whenArraysAreSameShape(self):
        data1 = [[1, 2, 3], [1, 2, 3,]]
        data2 = [[1, 2, 3], [1, 2, 3]]
        arr1 = np.array(data1)
        arr2 = np.array(data2)
        bsoid.check_arg.ensure_numpy_arrays_are_same_shape(arr1, arr2)

    def test__ensure_numpy_arrays_are_same_shape__shouldErrorOut__whenOneInputIsNotAnArray(self):
        data1 = [[1, 2, 3], [1, 2, 3]]
        data2 = [[1, 2, 3, 4], [1, 2, 3, 4]]
        arr1 = np.array(data1)
        list2 = data2
        expected_err = TypeError
        func = bsoid.check_arg.ensure_numpy_arrays_are_same_shape
        #     def assertRaises(self, expected_exception, *args, **kwargs):
        self.assertRaises(expected_err, func, arr1, list2)

    # Ensure type
    def test__ensure_type__shouldRunWithoutError__whenGivenSingularCorrectExpectedType(self):
        # Arrange
        integer_var = 1
        expected_type = int
        # Act, Assert
        try:
            bsoid.check_arg.ensure_type(integer_var, expected_type)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldRunWithoutError__whenGivenMultipleCorrectExpectedTypes(self):
        # Arrange
        integer_var = 1
        expected_types_tuple = (int, float)
        # Act, Assert
        try:
            bsoid.check_arg.ensure_type(integer_var, expected_types_tuple)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldRunWithoutError__whenGivenMultipleCorrectExpectedTypesAsStarArgs(self):
        # Arrange
        integer_var = 1
        expected_types_tuple = (int, float)
        # Act, Assert
        try:
            bsoid.check_arg.ensure_type(integer_var, *expected_types_tuple)
            self.assertTrue(True)  # TODO: low: is this line necessary for test to pass?
        except TypeError as te:
            raise te

    def test__ensure_type__shouldProduceError__whenGivenSingularIncorrectExpectedType(self):
        # Arrange
        integer_var = 1
        expected_type = float

        self.assertRaises(TypeError, bsoid.check_arg.ensure_type, integer_var, expected_type)

    def test__ensure_type__shouldProduceError__whenGivenMultipleIncorrectExpectedTypes(self):
        # Arrange
        integer_var = 1
        expected_type = (float, str)

        self.assertRaises(TypeError, bsoid.check_arg.ensure_type, integer_var, expected_type)

    def test__ensure_type__shouldProduceError__whenGivenSingularIncorrectExpectedTypeAsStarArgs(self):
        # Arrange
        integer_var = 1
        expected_type = (float, str)

        self.assertRaises(TypeError, bsoid.check_arg.ensure_type, integer_var, *expected_type)



    def test__(self):
        # Arrange

        # Act

        # Assert

        self.assertEqual(None, None)
        # self.assertRaises()

