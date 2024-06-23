import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import h5py


def load_dict_from_hdf5(filename):
    """
    Load a dictionary with string keys and NumPy array values from an HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file to load the data from.

    Returns:
    dict: Dictionary with string keys and NumPy array values.
    """
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def prepare_folds(
    embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/train_p2_2d.h5",
    train_data_path="../data/ready_data/train_pdb2272.csv",
    n_splits=5,
):
    # Load embeddings and process them
    embeddings = load_dict_from_hdf5(embedding_path)
    for key in embeddings:
        embeddings[key] = np.squeeze(embeddings[key])

    embed_df = pd.DataFrame(
        list(embeddings.items()), columns=["identifier", "embedding"]
    )

    # Load training data and merge with embeddings
    train_df = pd.read_csv(train_data_path)
    train_df = train_df.merge(embed_df, on="identifier")

    # Prepare data for GroupKFold
    X = train_df["sequence"].tolist()
    y = train_df["label"].tolist()
    groups = train_df["cluster"].tolist()
    gkf = GroupKFold(n_splits=n_splits)

    # Split data into training and validation folds
    train_folds = []
    valid_folds = []

    for train_idx, valid_idx in gkf.split(X, y, groups=groups):
        train = train_df.iloc[train_idx]
        valid = train_df.iloc[valid_idx]

        train_folds.append(train)
        valid_folds.append(valid)

    return train_folds[1], valid_folds[1]
