import argparse

import pandas as pd

from ..utils import (convert_fasta_to_df, filter_df, make_balanced_df,
                     prepare_neg_samples)


def merge_df(df_1: pd.DataFrame, df_2: pd.DataFrame):
    return df_1.merge(df_2, on="identifier")


def main(
) -> None:
    parser = argparse.ArgumentParser(description="Prepare training data from binders and non-binders FASTA files.")
    parser.add_argument("--binders_path", type=str, default="data/uniprot/go_0003677_swissprot.fasta",
                        help="Path to the FASTA file containing binders.")
    parser.add_argument("--non_binders_path", type=str, default="data/uniprot/notgo_0003723_notgo_0003677_swissprot.fasta",
                        help="Path to the FASTA file containing non-binders.")
    parser.add_argument("--path_yml", type=str, default="data/processed/neg_samples_annot.yml",
                        help="Path to the YML file")
    parser.add_argument("--output_path", type=str, default="data/embeddings/input_csv/train_p3.csv",
                        help="Path to save the prepared training data CSV.")
    args = parser.parse_args()

    binders = convert_fasta_to_df(args.binders_path)
    non_binders = convert_fasta_to_df(args.non_binders_path)

    neg_samples = prepare_neg_samples(args.path_yml)
    non_binders = merge_df(non_binders, neg_samples)

    non_binders["label"] = 0
    binders["label"] = 1

    train = pd.concat([binders, non_binders])
    train = filter_df(train)
    train = make_balanced_df(train)
    train.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
