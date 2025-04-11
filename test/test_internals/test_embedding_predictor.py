import unittest

import numpy.testing as np_test
import pandas as pd
from sklearn.metrics import r2_score

from biked_commons.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from biked_commons.bike_embedding.embedding_comparator import get_cosine_similarity
from biked_commons.bike_embedding.embedding_predictor import EmbeddingPredictor
from test.utils_for_tests import path_of_test_resource


class EmbeddingPredictorTest(unittest.TestCase):
    def setUp(self):
        self.embedding_predictor = EmbeddingPredictor()
        self.embeddings = pd.read_csv(path_of_test_resource("subset_embeddings.csv"), index_col=0)
        self.parameters = pd.read_csv(path_of_test_resource("subset_parametric.csv"), index_col=0)

    def test_scaled_r2(self):
        calc = ClipEmbeddingCalculatorImpl()
        target = calc.from_text("A green bicycle with thick wheels")
        score = r2_score(get_cosine_similarity(self.embeddings.values, target),
                         get_cosine_similarity(self.embedding_predictor.predict(self.parameters),
                                               target))
        self.assertGreater(score, 0.93)

    def test_scaled_predict_one(self):
        self.embedding_predictor.predict(self.parameters.iloc[0:1])

    def test_shape(self):
        predictions = self.embedding_predictor.predict(self.parameters)
        self.assertEqual(predictions.shape, (len(self.parameters), 512))

    def test_order_does_not_matter(self):
        data = self.parameters.iloc[0:1]
        predictions_og = self.embedding_predictor.predict(data)
        first_column = list(data.columns)[0]
        modified_data = data.drop(labels=[first_column], axis=1)
        modified_data[first_column] = data[first_column]
        reordered_predictions = self.embedding_predictor.predict(modified_data)
        np_test.assert_array_equal(
            predictions_og,
            reordered_predictions
        )

    @unittest.skip
    def test_handles_nan(self):
        pass
