"""
Create tests specifically for the PipelinePrime object
"""
from typing import Set
from unittest import TestCase, skip
import os

import bsoid

long_csv_test_file_path = "FullSample_Video1DLC_resnet50_EPM_DLC_BSOIDAug25shuffle1_495000.csv"
csv_test_file_path = os.path.join(bsoid.config.BSOID_BASE_PROJECT_PATH, 'tests', 'test_data', long_csv_test_file_path)
assert os.path.isfile(csv_test_file_path)


class TestPipeline(TestCase):

    @skip
    def test__(self):
        pass
