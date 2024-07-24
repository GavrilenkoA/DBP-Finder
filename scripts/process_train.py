import pandas as pd
import yaml
from utils import convert_fasta_to_df, filter_df, make_balanced_df


def prepare_neg_samples() -> pd.DataFrame:
    with open("data/processed/neg_samples_annot.yml", "r") as file:
        neg_samples_annotation = yaml.safe_load(file)

    identifiers = list()
    for protein_id in neg_samples_annotation:
        if not neg_samples_annotation[protein_id]:
            identifiers.append(protein_id)

    df = pd.DataFrame(list(identifiers), columns=["identifier"])
    return df


def merge_neg_samples(df_1: pd.DataFrame, df_2: pd.DataFrame):
    return df_1.merge(df_2, on="identifier")


def prepare_training_data(
    binders_path: str = "data/uniprot/go_0003677_swissprot.fasta",
    non_binders_path: str = "data/uniprot/notgo_0003723_notgo_0003677_swissprot.fasta",
    output_path: str = "data/embeddings/input_csv/train_p3.csv",
) -> None:
    """
    Prepares and saves the training data by processing binders and non-binders from given FASTA files.

    Args:
        binders_path (str): Path to the FASTA file containing binders.
        non_binders_path (str): Path to the FASTA file containing non-binders.
        output_path (str): Path to save the prepared training data CSV.
    """
    binders = convert_fasta_to_df(binders_path)
    binders["label"] = 1

    non_binders = convert_fasta_to_df(non_binders_path)
    non_binders["label"] = 0

    neg_samples = prepare_neg_samples()
    non_binders = merge_neg_samples(non_binders, neg_samples)

    train = pd.concat([binders, non_binders])
    train = filter_df(train)
    train = make_balanced_df(train)
    train.to_csv(output_path, index=False)


if __name__ == "__main__":
    prepare_training_data()
