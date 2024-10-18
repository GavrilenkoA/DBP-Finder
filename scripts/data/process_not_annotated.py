import pandas as pd
from ..utils import convert_fasta_to_df, filter_df, prepare_neg_samples, write_fasta


def filter_not_annotated_df(not_annotated_df, train) -> pd.DataFrame:
    # Exclude identifiers based on merge with train on 'identifier'
    identfier_exclude = not_annotated_df.merge(train, on="identifier")["identifier"]
    not_annotated_df = not_annotated_df[
        ~not_annotated_df["identifier"].isin(identfier_exclude)
    ]

    # Exclude identifiers based on merge with train on 'sequence'
    identfier_exclude = not_annotated_df.merge(train, on="sequence")["identifier_x"]
    not_annotated_df = not_annotated_df[
        ~not_annotated_df["identifier"].isin(identfier_exclude)
    ]
    return not_annotated_df


def main(
    path_fasta: str = "data/not_annotated/raw_fasta/merged.fasta",
    train_path: str = "data/embeddings/input_csv/train_p3.csv",
    output_path: str = "data/not_annotated/not_annotated.csv",
) -> None:

    # neg_samples_df = prepare_neg_samples("data/not_annotated/not-annotated-GO:0003676.yml")
    not_annotated_df = convert_fasta_to_df(path_fasta)
    not_annotated_df = filter_df(not_annotated_df)
    # not_annotated_df = neg_samples_df.merge(not_annotated_df, on="identifier")

    # train = pd.read_csv(train_path)
    # not_annotated_df = filter_not_annotated_df(not_annotated_df, train)
    # write_fasta(not_annotated_df, "not-annotated-GO:0003676.fasta")

    not_annotated_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
