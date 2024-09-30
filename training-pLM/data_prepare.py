import json
import h5py
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_dict_from_hdf5(filename):
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def make_folds(
    df: pd.DataFrame, n_splits: int = 5
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    # Prepare data for GroupKFold
    X = df["sequence"]
    y = df["label"]
    groups = df["cluster"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    # Split data into training and validation folds
    train_folds = []
    valid_folds = []

    for train_idx, valid_idx in sgkf.split(X, y, groups=groups):
        train = df.iloc[train_idx]
        valid = df.iloc[valid_idx]

        train_folds.append(train)
        valid_folds.append(valid)

    return train_folds, valid_folds


def load_embeddings_to_df(embedding_path: str) -> pd.DataFrame:
    # Load embeddings from the HDF5 file
    embeddings = load_dict_from_hdf5(embedding_path)

    # Convert the embeddings dictionary to a DataFrame
    embeddings_df = pd.DataFrame(
        list(embeddings.items()), columns=["identifier", "embedding"]
    )
    return embeddings_df


def get_embed_clustered_df(
    embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/train_p2_2d.h5",
    csv_path="../data/splits/train_p2.csv",
) -> pd.DataFrame:
    # Load training data and merge with embeddings
    df = pd.read_csv(csv_path)
    embeddings_df = load_embeddings_to_df(embedding_path)

    embed_df = df.merge(embeddings_df, on="identifier")
    assert len(embed_df) == len(df), "embed_df and df have different lengths"
    return embed_df


def prepare_test(
    df: pd.DataFrame,
    embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/train_p2_2d.h5",
) -> pd.DataFrame:
    embeddings = load_dict_from_hdf5(embedding_path)
    embeddings_df = pd.DataFrame(
        list(embeddings.items()), columns=["identifier", "embedding"]
    )

    embed_df = df.merge(embeddings_df, on="identifier")
    assert len(embed_df) == len(df), "embed_df and df have different lengths"
    return embed_df


def form_test_kingdom(test_input: str, kingdom: str):
    kingdom_df = pd.read_csv(f"../data/taxon/{test_input}.csv")
    kingdom_df = kingdom_df[kingdom_df["kingdom"] == kingdom]

    test_df = pd.read_csv(f"../data/embeddings/input_csv/{test_input}.csv")
    test_df = test_df.merge(kingdom_df, on="identifier")
    return test_df


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
