from typing import Generator

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from utils import SEED


def stratified_group_kfold_split(
    data_path: str = "data/rna/processed/train_clustered.csv",
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    # Read the data
    train = pd.read_csv(data_path)

    # Extract features and labels
    X = train["sequence"]
    y = train["label"]
    groups = train["cluster"]

    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5)

    # Iterate over folds
    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        train_sample = train.iloc[train_idx]
        test_sample = train.iloc[test_idx]
        yield train_sample, test_sample


def train_test_split_and_merge(
    data_path: str = "data/rna/processed/train_clustered.csv",
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read the data
    train = pd.read_csv(data_path)
    # Perform train-test split based on identifier
    X = train["identifier"]
    y = train["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Create DataFrame for train and test sets
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Merge cluster information into train and test sets
    clusters = train.loc[:, ["identifier", "cluster"]]
    df_train = df_train.merge(clusters, on="identifier")
    df_test = df_test.merge(clusters, on="identifier")

    return df_train, df_test
