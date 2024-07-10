import argparse
import os

import pandas as pd
from rna_process_data import load_test_rna_datasets
from train_test_cluster import cluster_data
from utils import make_balanced_df, reduce_train


def make_redundant_train(path_train: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path_train)
    test = load_test_rna_datasets()

    output_mmseq = cluster_data(train, test)
    clustered_train, clustered_test = reduce_train(output_mmseq)

    train_ = train.merge(clustered_train, on="identifier")
    train_ = make_balanced_df(train_)

    test_ = test.merge(clustered_test, on="identifier")
    assert len(test_) == len(test), "Lost test sequences"

    return train_, test_


def make_single_for_cluster_train(train, input_test):
    grouped = train.groupby("cluster")

    single_df = []
    for _, group in grouped:
        single_df.append(group.iloc[0:1])
    single_df = pd.concat(single_df)

    train_path = f"data/ready_data/train_{input_test}_single.csv"
    if not os.path.exists(train_path):
        single_df.to_csv(train_path, index=False)

    single_df = make_balanced_df(single_df)

    train_path = f"data/ready_data/train_{input_test}_single_balanced.csv"
    if not os.path.exists(train_path):
        single_df.to_csv(train_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Program\
                                    for clustering protein sequences"
    )
    parser.add_argument("path_train", type=str, help="A string argument")
    args = parser.parse_args()

    train, test = make_redundant_train(args.path_train)

    train.to_csv("data/rna/splits/train.csv", index=False)
    test.to_csv("data/rna/splits/test.csv", index=False)


if __name__ == "__main__":
    main()
