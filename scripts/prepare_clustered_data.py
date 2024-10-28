import argparse
import os

import pandas as pd

from train_test_cluster import cluster_data
from utils import make_balanced_df, reduce_train


def main():
    parser = argparse.ArgumentParser(
        description="Program\
                                    for clustering protein sequences"
    )
    parser.add_argument("path_train", type=str, help="Path to the train CSV file")
    parser.add_argument("path_test", type=str, help="Path to the test CSV file")
    args = parser.parse_args()

    train = pd.read_csv(args.path_train)
    test = pd.read_csv(args.path_test)

    output_mmseq = cluster_data(train, test)
    clustered_train, _ = reduce_train(output_mmseq)
    train = train.merge(clustered_train, on="identifier")
    train = make_balanced_df(train)

    name_train = os.path.splitext(os.path.basename(args.path_train))[0]
    name_test = os.path.splitext(os.path.basename(args.path_test))[0]
    output_csv = f"data/splits/{name_train}_{name_test}__.csv"
    train.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
