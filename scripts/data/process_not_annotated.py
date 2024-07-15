import pandas as pd


def filter_and_save_not_annotated(
    not_annotated_path: str = "data/not_annotated/not_annotated_seqs_v0.csv",
    train_path: str = "data/embeddings/input_csv/train_p2.csv",
    output_path: str = "data/not_annotated/not_annotated_seqs_v1.csv",
) -> None:
    not_annotated_df = pd.read_csv(not_annotated_path)
    train = pd.read_csv(train_path)

    identfier_exclude = not_annotated_df.merge(train, on="identifier")["identifier"]
    not_annotated_df = not_annotated_df[
        ~not_annotated_df["identifier"].isin(identfier_exclude)
    ]

    identfier_exclude = not_annotated_df.merge(train, on="sequence")["identifier_x"]
    not_annotated_df = not_annotated_df[
        ~not_annotated_df["identifier"].isin(identfier_exclude)
    ]

    not_annotated_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    filter_and_save_not_annotated()
