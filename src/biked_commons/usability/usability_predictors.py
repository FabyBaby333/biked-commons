import dill
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from biked_commons.resource_utils import resource_path
from biked_commons.usability.mlp_model import MLP
from biked_commons.usability.ordered_columns import ORDERED_COLUMNS


_SVM_MODEL_PATH = resource_path("svm_model.pkl")
_MLP_MODEL_PATH = resource_path("mlp_with_hyperparameters.pth")
_DATA_PATH = resource_path("datasets/raw_datasets/clip_sBIKED_processed.csv")
_SCALER_PATH = resource_path("scaler_usability.pk")


def _load_scaler() -> preprocessing.MinMaxScaler:
    try:
        with open(_SCALER_PATH, "rb") as file:
            return dill.load(file)
    except FileNotFoundError:
        df = pd.read_csv(_DATA_PATH, index_col=0)[ORDERED_COLUMNS]
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df.values)
        with open(_SCALER_PATH, "wb") as file:
            dill.dump(scaler, file)
        return scaler


def _load_svm_model():
    return joblib.load(_SVM_MODEL_PATH)


def _load_mlp_model():
    checkpoint = torch.load(_MLP_MODEL_PATH)
    hp = checkpoint["hyperparameters"]
    model = MLP(
        input_dim=hp["input_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout_rate=hp["dropout_rate"],
        lr=hp["lr"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class UsabilityPredictorBinary:
    """Predicts binary usability. Works on unscaled input."""

    def __init__(self):
        self._scaler = _load_scaler()
        self._model = _load_svm_model()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._predict(self._inverse_scale_if_needed(self._get_ordered(x)))

    def _get_ordered(self, x: pd.DataFrame):
        return pd.DataFrame(x, columns=ORDERED_COLUMNS)

    def _inverse_scale_if_needed(self, x_df: pd.DataFrame):
        x_values = x_df.values
        if x_values.min() >= 0 and x_values.max() <= 1:
            return pd.DataFrame(self._scaler.inverse_transform(x_values), columns=ORDERED_COLUMNS)
        return x_df

    def _predict(self, x_df: pd.DataFrame) -> np.ndarray:
        return self._model.predict(x_df.values)


class UsabilityPredictorContinuous:
    """Predicts continuous usability. Works on scaled input."""
    def __init__(self):
        self._scaler = _load_scaler()
        self._model = _load_mlp_model()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._predict(self._scale_if_needed(self._get_ordered(x)))

    def _get_ordered(self, x: pd.DataFrame):
        return pd.DataFrame(x, columns=ORDERED_COLUMNS)

    def _scale_if_needed(self, x_df: pd.DataFrame):
        x_values = x_df.values
        if x_values.min() < 0 or x_values.max() > 1:
            return pd.DataFrame(self._scaler.transform(x_values), columns=ORDERED_COLUMNS)
        return x_df

    def _predict(self, x_df: pd.DataFrame) -> np.ndarray:
        tensor = torch.tensor(x_df.values, dtype=torch.float32)
        with torch.no_grad():
            return self._model(tensor).numpy()
