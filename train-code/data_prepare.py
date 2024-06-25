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


def make_folds(
    df: pd.DataFrame, n_splits: int = 5
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:

    # Prepare data for GroupKFold
    X = df["sequence"].tolist()
    y = df["label"].tolist()
    groups = df["cluster"].tolist()
    gkf = GroupKFold(n_splits=n_splits)

    # Split data into training and validation folds
    train_folds = []
    valid_folds = []

    for train_idx, valid_idx in gkf.split(X, y, groups=groups):
        train = df.iloc[train_idx]
        valid = df.iloc[valid_idx]

        train_folds.append(train)
        valid_folds.append(valid)

    return train_folds, valid_folds


def prepare_embed_df(
    embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/train_p2_2d.h5",
    csv_path="../data/ready_data/train_pdb2272.csv",
) -> pd.DataFrame:

    # Load embeddings and process them
    embeddings = load_dict_from_hdf5(embedding_path)
    for key in embeddings:
        embeddings[key] = np.squeeze(embeddings[key])

    embeddings_df = pd.DataFrame(
        list(embeddings.items()), columns=["identifier", "embedding"]
    )

    # Load training data and merge with embeddings
    df = pd.read_csv(csv_path)
    embed_df = df.merge(embeddings_df, on="identifier")
    assert len(embed_df) == len(df), "embed_df and df have different lengths"
    return embed_df
