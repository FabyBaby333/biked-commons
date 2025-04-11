import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.core.common import flatten

from biked_commons.api.validations import validate_raw_BikeCad, validate_clip
from biked_commons.resource_utils import resource_path
from biked_commons.validation.clip_validation_functions import CLIPS_VALIDATIONS
from biked_commons.validation.raw_validation_functions import RAW_VALIDATION_FUNCTIONS
from biked_commons.xml_handling.algebraic_parser import AlgebraicParser
from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler
from test.utils_for_tests import path_of_test_resource


class ValidationsTest(unittest.TestCase):
    def setUp(self):
        with open(path_of_test_resource("clip_sBIKED_processed.csv"), "r") as file:
            self.df = pd.read_csv(file)

    def test_validate_clip(self):
        validation_results = validate_clip(self.df)
        expected_rows = len(self.df)
        expected_columns = len(CLIPS_VALIDATIONS)
        self.assertEqual((expected_rows, expected_columns), validation_results.shape)

    def test_validate_biked(self):
        data_length = 31
        needed_columns = [c for c in flatten(
            [validation_function.variable_names() for validation_function in RAW_VALIDATION_FUNCTIONS])]
        data = self._generate_data(data_length)
        data = data.drop(labels=[c for c in data.columns if c not in needed_columns], axis=1).astype("float32")
        validation_results = validate_raw_BikeCad(data)
        expected_shape = (data_length, len(RAW_VALIDATION_FUNCTIONS))
        self.assertEqual(expected_shape, validation_results.shape)

    def test_exceptional_cases_have_correct_dimensions(self):
        data_length = 31
        results = validate_raw_BikeCad(pd.DataFrame.from_records({"FIELD": 15} for _ in range(data_length)))
        all_invalid = np.ones(shape=(31, len(RAW_VALIDATION_FUNCTIONS)))
        np_test.assert_array_equal(all_invalid, results.to_numpy())


    def _generate_data(self, data_length):
        handler = BikeXmlHandler()
        with open(resource_path("PlainRoadBikeStandardized.txt"), "r") as file:
            handler.set_xml(file.read())
        data = pd.DataFrame.from_records([self._to_dict_data(handler) for _ in range(data_length)])
        return data

    def _to_dict_data(self, handler):
        res = {k: AlgebraicParser().attempt_parse(v) for k, v in handler.get_entries_dict().items()}
        res["DT Length"] = 600
        return res
