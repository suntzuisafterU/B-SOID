"""
Create tests specifically for the PipelinePrime object
"""
from typing import Set
from unittest import TestCase, skip
import os
import random

import bsoid

long_csv_test_file_path = "FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
csv_test_file_path = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', long_csv_test_file_path)
csv_test_file_path = bsoid.config.DEFAULT_PIPELINE__PRIME__CSV_TEST_FILE_PATH
assert os.path.isfile(csv_test_file_path)


class TestPipeline(TestCase):

    @skip  # Temporarily skipped since the test data (10 rows) isn't enough to pull out more than 1 cluster
    def test___CanItEvenBuild(self):
        # Arrange
        gmm_n_components, cv = 2, 3  # Set gmm clusters low so that it can still work with 10 rows of data
        p = bsoid.pipeline.PipelinePrime(f'TestPipeline_{random.randint(0, 100_000_000)}',
                                         cross_validation_k=cv,
                                         gmm_n_components=gmm_n_components,
                                         )
        err = f"""Sanity Check: Something bad happened and cross val is not right"""
        self.assertEqual(cv, p.cross_validation_k, err)
        p = p.add_train_data_source(csv_test_file_path)
        # Act

        p = p.build()
        # Assert
        self.assertTrue(True)




