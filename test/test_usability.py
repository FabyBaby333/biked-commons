import unittest
import numpy as np
import pandas as pd
import biked_commons.prediction.evaluators
import biked_commons.prediction.loaders
from biked_commons.usability.usability_predictors import UsabilityPredictorBinary, UsabilityPredictorContinuous

loader = biked_commons.prediction.loaders
evaluator = biked_commons.prediction.evaluators.evaluate_usability

class BinaryUsabilityTest(unittest.TestCase):
    
    def setUp(self):
        self.predictor = UsabilityPredictorBinary()
        self.X_train, self.Y_train, self.X_test = loader.load_usability('binary')
    
    def test_data_loading(self):
        self.assertIsNotNone(self.X_train)
        self.assertIsNotNone(self.Y_train)
        self.assertIsNotNone(self.X_test)
        self.assertEqual(len(self.X_train), len(self.Y_train))
    
    def test_prediction_shape(self):
        preds = self.predictor.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
    
    def test_prediction_values(self):
        preds = self.predictor.predict(self.X_test)
        unique_vals = np.unique(preds)
        for val in unique_vals:
            self.assertIn(val, [0, 1], msg=f"Unexpected prediction value: {val}")
    
    def test_evaluator_output(self):
        preds = self.predictor.predict(self.X_test)
        result = evaluator(preds, 'binary')
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_evaluator_output_scaled(self):
        data = {'Saddle height': [0.45104834], 'Stack': [0.4110677], 'CS textfield': [0.215]}
        data_df = pd.DataFrame(data)
        preds = self.predictor.predict(data_df)
        self.assertGreaterEqual(preds[0], 0.0)
        self.assertLessEqual(preds[0], 1.0)

    def test_input_not_dataframe(self):
        data = [[0.45104834, 0.4110677, 0.215]]
        preds = self.predictor.predict(data)
        self.assertGreaterEqual(preds[0], 0.0)
        self.assertLessEqual(preds[0], 1.0)
    

class ContinuousUsabilityTest(unittest.TestCase):
    
    def setUp(self):
        self.predictor = UsabilityPredictorContinuous()
        self.X_train, self.Y_train, self.X_test = loader.load_usability('cont')
    
    def test_data_loading(self):
        self.assertIsNotNone(self.X_train)
        self.assertIsNotNone(self.Y_train)
        self.assertIsNotNone(self.X_test)
        self.assertEqual(len(self.X_train), len(self.Y_train))

    def test_prediction_shape(self):
        preds = self.predictor.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])
    
    def test_prediction_values(self):
        preds = self.predictor.predict(self.X_test)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating), msg=f"Predictions are not of float type, found {preds.dtype}")

    def test_evaluator_output_scaled(self):
        preds = self.predictor.predict(self.X_test)
        result = evaluator(preds, 'cont')
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)
        
    def test_evaluator_output_nonscaled(self):
        data = {'Saddle height': [768.0], 'Stack': [565.6], 'CS textfield': [430]}
        data_df = pd.DataFrame(data)
        preds = self.predictor.predict(data_df)
        self.assertGreaterEqual(preds[0], 0.0)
        self.assertLessEqual(preds[0], 1.0)

    def test_input_not_dataframe(self):
        data = [[768.0,565.6,430]]
        preds = self.predictor.predict(data)
        self.assertGreaterEqual(preds[0], 0.0)
        self.assertLessEqual(preds[0], 1.0)