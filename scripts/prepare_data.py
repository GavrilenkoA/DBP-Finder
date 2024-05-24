import pandas as pd
import os
from cluster_sequences import cluster_data
from utils import reduce_train, make_balanced_df


def make_redundant_train(input_test):
    train = pd.read_csv("data/embeddings/input_csv/train_p2.csv")
    test = pd.read_csv(f"data/embeddings/input_csv/{input_test}.csv")

    output_mmseq = cluster_data(train, test, identity=0.5)
    output_mmseq = reduce_train(output_mmseq)

    train["identifier"] = train["identifier"].apply(lambda x: x.split("_")[0])
    train = train.merge(output_mmseq, on="identifier")
    train = make_balanced_df(train)

    train_path = f"data/ready_data/train_{input_test}.csv"
    if not os.path.exists(train_path):
        train.to_csv(train_path, index=False)

    return train


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
    input_test = input()

    train = make_redundant_train(input_test)
    make_single_for_cluster_train(train, input_test)


if __name__ == "__main__":
    main()
