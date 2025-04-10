import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, r2_score, mean_squared_error, accuracy_score

from biked_commons.resource_utils import resource_path

def evaluate_usability(predictions, target_type):
    if target_type == 'cont':
        Y_test = pd.read_csv(resource_path('datasets/split_datasets/usability_cont_Y_test.csv'), index_col=0)
        return r2_score(Y_test, predictions)
    
    elif target_type == 'binary':
        Y_test = pd.read_csv(resource_path('datasets/split_datasets/usability_binary_Y_test.csv'), index_col=0)
        return accuracy_score(Y_test, predictions)
    else:
        raise ValueError("Invalid target_type. Choose either 'cont' or 'binary'.")

def evaluate_frame_validity(predictions):
    Y_test = pd.read_csv(resource_path('datasets/split_datasets/validity_Y_test.csv'), index_col=0)
    return f1_score(Y_test, predictions)

def evaluate_structure(predictions):
    Y_test = pd.read_csv(resource_path('datasets/split_datasets/structure_Y_test.csv'), index_col=0)
    return r2_score(Y_test, predictions)

def evaluate_aero(predictions):
    Y_test = pd.read_csv(resource_path('datasets/split_datasets/aero_Y_test.csv'), index_col=0)
    return r2_score(Y_test, predictions)

def evaluate_clip(predictions):
    Y_test = np.load(resource_path('datasets/raw_datasets/CLIP_Y_test.npy'))
    return mean_squared_error(Y_test, predictions)
