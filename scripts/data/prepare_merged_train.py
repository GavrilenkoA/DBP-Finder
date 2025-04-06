import argparse

import pandas as pd

from ..utils import filter_df


def prepare_other_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = filter_df(df, min_len=50 * 1 / 4, max_len=1024 * 4)
    df = df.drop(columns=["label"], axis=1)
    return df


def load_my_train(path: str = "data/embeddings/input_csv/train_p3.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["label"], axis=1)
    return df


def merge_df(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    df_1["identifier"] = df_1["identifier"].apply(lambda x: str(x) + "one")
    df_2["identifier"] = df_2["identifier"].apply(lambda x: x + "two")
    df = pd.concat([df_1, df_2])
    df = df.drop_duplicates(subset=["sequence"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data from binders and non-binders FASTA files.")
    parser.add_argument("train_path", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    df_1 = prepare_other_train(args.train_path)
    df_2 = load_my_train()
    df = merge_df(df_1, df_2)
    df.to_csv(f"data/processed/{args.output_file}_train_p3_merged.csv", index=False)


if __name__ == "__main__":
    main()
