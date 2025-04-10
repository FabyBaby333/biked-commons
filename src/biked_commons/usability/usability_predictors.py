import joblib
import pandas as pd
import torch
from sklearn import preprocessing

from biked_commons.resource_utils import resource_path
from biked_commons.usability.mlp_model import MLP


_SVM_MODEL_WEIGHTS_PATH = resource_path("svm_model.pkl")
_MLP_MODEL_WEIGHTS_PATH = resource_path("mlp_with_hyperparameters.pth")
_CLIP_BIKED_PROCESSED_PATH = resource_path("datasets/raw_datasets/clip_sBIKED_processed.csv")
_SCALER_PATH = resource_path("scaler_usability.pk")

class UsabilityPredictorBinary:
    def __init__(self):
        self.features = ['Saddle height', 'Stack', 'CS textfield']
        self.model = joblib.load(_SVM_MODEL_WEIGHTS_PATH)

        self.min_max_scaler = self._get_scaler()

    def _get_scaler(self):
        try:
            return joblib.load(_SCALER_PATH) 
        except FileNotFoundError:
            df = pd.read_csv(_CLIP_BIKED_PROCESSED_PATH, index_col=0)
            df = df[self.features]
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.values)
            joblib.dump(scaler, _SCALER_PATH)
            return scaler

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x_filtered = x[self.features].values
        else:
            # If it's not a DataFrame, assume it's already the correct input format
            x_filtered = x

        
        # If the data is scaled, inverse scale it
        if x_filtered.min() >= 0 and x_filtered.max() <= 1:
            x_filtered = self.min_max_scaler.inverse_transform(x_filtered)

        return self.model.predict(x_filtered)



class UsabilityPredictorContinuous:
    def __init__(self):
        self.features = ['Saddle height', 'Stack', 'CS textfield']
        checkpoint = torch.load(_MLP_MODEL_WEIGHTS_PATH)
        hyperparameters = checkpoint['hyperparameters']

        self.model = MLP(
            input_dim=hyperparameters["input_dim"],
            hidden_dims=hyperparameters["hidden_dims"],
            dropout_rate=hyperparameters["dropout_rate"],
            lr=hyperparameters["lr"]
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.min_max_scaler = self._get_scaler()

    def _get_scaler(self):
        try:
            return joblib.load(_SCALER_PATH) 
        except FileNotFoundError:
            df = pd.read_csv(_CLIP_BIKED_PROCESSED_PATH, index_col=0)
            df = df[self.features]
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(df.values)
            joblib.dump(scaler, _SCALER_PATH)
            return scaler

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x_filtered = x[self.features].values
        else:
            # If it's not a DataFrame, assume it's already the correct input format
            x_filtered = x

        # If not scaled, scale it
        if x_filtered.min() < 0 or x_filtered.max() > 1:
            x_filtered = self.min_max_scaler.transform(x_filtered)
            
        self.model.eval()
        with torch.no_grad(): 
            x_tensor = torch.tensor(x_filtered, dtype=torch.float32)
            return self.model(x_tensor).numpy() 

