import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
from tqdm import tqdm

from biked_commons.resource_utils import resource_path


def prepare_validity():
    df = pd.read_csv(resource_path('datasets/raw_datasets/validity.csv'), index_col=0)
    df = df.reset_index(drop=True)
    subset = df[df['valid'].isin([0, 2])]
    Y = subset['valid']
    X = subset.drop('valid', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    X_train.to_csv(resource_path('datasets/split_datasets/validity_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/validity_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/validity_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/validity_Y_test.csv'))

def prepare_structure():
    df = pd.read_csv(resource_path('datasets/raw_datasets/structure.csv'), index_col=0)
    df = df.reset_index(drop=True)

    sim_1_displacements = df[["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp."]].values
    sim_1_abs_displacements = np.abs(sim_1_displacements)
    sim_1_normalized_displacements = sim_1_abs_displacements / np.mean(sim_1_abs_displacements, axis=0)
    sim_1_compliance_score = np.mean(sim_1_normalized_displacements, axis=1)

    sim_2_displacements = df["Sim 2 Bottom Bracket Z Disp."].values
    sim_2_abs_displacements = np.abs(sim_2_displacements)
    sim_2_compliance_score = sim_2_abs_displacements / np.mean(sim_2_displacements)

    sim_3_displacements = df[["Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot."]].values
    sim_3_abs_displacements = np.abs(sim_3_displacements)
    sim_3_normalized_displacements = sim_3_abs_displacements / np.mean(sim_3_abs_displacements, axis=0)
    sim_3_compliance_score = np.mean(sim_3_normalized_displacements, axis=1)

    mass = df["Model Mass"].values
    planar_SF = df["Sim 1 Safety Factor"].values
    eccentric_SF = df["Sim 3 Safety Factor"].values

    Y = np.stack([mass, sim_1_compliance_score, sim_2_compliance_score, sim_3_compliance_score, planar_SF, eccentric_SF], axis=1)
    Y = pd.DataFrame(Y, columns=["Mass", "Planar Compliance", "Transverse Compliance", "Eccentric Compliance", "Planar Safety Factor", "Eccentric Safety Factor"])
    X = df.drop(["Model Mass", "Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot.", "Sim 1 Safety Factor", "Sim 3 Safety Factor"], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv(resource_path('datasets/split_datasets/structure_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/structure_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/structure_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/structure_Y_test.csv'))

def prepare_aero():
    df = pd.read_csv(resource_path('datasets/raw_datasets/aero.csv'), index_col=0)
    df = df.reset_index(drop=True)

    Y = df["drag"]
    X = df.drop("drag", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv(resource_path('datasets/split_datasets/aero_X_train.csv'))
    X_test.to_csv(resource_path('datasets/split_datasets/aero_X_test.csv'))
    Y_train.to_csv(resource_path('datasets/split_datasets/aero_Y_train.csv'))
    Y_test.to_csv(resource_path('datasets/split_datasets/aero_Y_test.csv'))

def download_file(file_url, file_path):
    """Downloads a file with a progress bar if it doesn't exist locally."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists

    response = requests.get(file_url, stream=True)
    
    if response.status_code == 200:
        file_size = int(response.headers.get('content-length', 0))  # Get file size if available
        chunk_size = 1024  # 1 KB per chunk

        with open(file_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(file_path)}",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"✅ Download complete: {file_path}")
    else:
        print(f"❌ Error downloading {file_path}: {response.status_code}")

def check_download_CLIP_data():
    # File paths and URLs
    x_id = "10541435"
    x_url = f"https://dataverse.harvard.edu/api/access/datafile/{x_id}"
    x_file = resource_path('datasets/split_datasets/CLIP_X_train.csv')

    y_id = "10992683"
    y_url = f"https://dataverse.harvard.edu/api/access/datafile/{y_id}"
    y_file = resource_path('datasets/split_datasets/CLIP_Y_train.npy')

    # Check and download X
    if not os.path.exists(x_file):
        print(f"⚠️  {os.path.basename(x_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(x_url, x_file)
    else:
        print(f"✅ {os.path.basename(x_file)} already exists in datasets folder. Skipping download.")

    # Check and download Y
    if not os.path.exists(y_file):
        print(f"⚠️  {os.path.basename(y_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(y_url, y_file)
    else:
        print(f"✅ {os.path.basename(y_file)} already exists in datasets folder. Skipping download.")

def prepare_clip():
    check_download_CLIP_data()

