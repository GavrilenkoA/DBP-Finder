import argparse
import os

import pandas as pd
from train_test_cluster import cluster_data
from utils import RNADataset, make_balanced_df, reduce_train


def make_redundant_train(path_train: str, path_test: str):
    train = pd.read_csv(path_train)

    test_dataset = RNADataset(path_test)
    test = test_dataset.get_data()

    output_mmseq = cluster_data(train, test)
    reduced_train = reduce_train(output_mmseq)

    train = train.merge(reduced_train, on="identifier")
    train = make_balanced_df(train)

    return train, test


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
    # parser = argparse.ArgumentParser(description="Program\
    #                                 for clustering protein sequences")
    # parser.add_argument("path_train", type=str, help="A string argument")
    # parser.add_argument("path_test", type=str, help="A string argument")

    # args = parser.parse_args()
    train, test = make_redundant_train(
        "data/rna/processed/train.csv",
        "data/rna/raw/9606_accending_trP1170_trN8485_VaP126_VaN942_TeP178_TeN1202_pep_label.csv",
    )
    train.to_csv("ex.csv", index=False)


if __name__ == "__main__":
    main()
