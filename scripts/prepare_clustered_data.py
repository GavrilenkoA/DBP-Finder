import argparse
import os

import pandas as pd
from train_test_cluster import cluster_data
from utils import make_balanced_df, reduce_train, write_fasta


# train_ = make_balanced_df(train_)
# test_ = test.merge(clustered_test, on="identifier")
# assert len(test_) == len(test), "Lost test sequences"


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
    parser.add_argument("path_test", type=str, help="A string argument")
    args = parser.parse_args()

    train = pd.read_csv(args.path_train)
    test = pd.read_csv(args.path_test)

    output_mmseq = cluster_data(train, test)
    clustered_train, _ = reduce_train(output_mmseq)
    train = train.merge(clustered_train, on="identifier")
    train = make_balanced_df(train)

    name_train = os.path.splitext(os.path.basename(args.path_train))[0]
    name_test = os.path.splitext(os.path.basename(args.path_test))[0]
    output_csv = f"data/splits/{name_train}_{name_test}.csv"
    train.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
