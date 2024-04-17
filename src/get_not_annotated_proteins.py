import logging
import requests
import pandas as pd
from tqdm import tqdm


def collect_data(fasta_file: str, pos: int = 1) -> tuple[list, list]:
    with open(fasta_file) as fi:
        lines = fi.readlines()

    identifiers = []
    sequences = []
    seq = ""

    for line in lines:
        if line.startswith(">"):
            head = line.split("|")[pos].replace(">", "")
            identifiers.append(head)
            if seq:
                sequences.append(seq)
                seq = ""
        else:
            seq += line.strip()

    if seq:
        sequences.append(seq)
    assert len(identifiers) == len(sequences)
    return identifiers, sequences


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df.loc[df["sequence"].apply(valid_sequence)]
    df = df.loc[df["sequence"].apply(lambda x: 49 < len(x) < 1025)]
    df = df.drop_duplicates(subset=["sequence"])
    return df


def have_annotation(id_protein: str) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    response = requests.get(url)
    annotation_data = response.json()

    annotation = False
    if annotation_data["numberOfHits"]:
        for item in annotation_data["results"]:
            if item["goId"] in {"GO:0003677", "GO:0003723"}:
                annotation = True
                break
    return annotation
# def have_annotation(id_protein: str) -> bool:
#     url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
#     response = requests.get(url)
#     annotation_data = response.json()
#     annotation = False
#     if annotation_data["numberOfHits"]:
#         annotation = True
#     return annotation


def write_not_annotated_seqs(identifiers: list, sequences: list) -> pd.DataFrame:
    identifiers_ = []
    sequences_ = []

    for id_, seq in tqdm(zip(identifiers, sequences), total=len(identifiers)):
        if not have_annotation(id_):
            identifiers_.append(id_)
            sequences_.append(seq)
        else:
            logging.info(f"{id_}")

    df = pd.DataFrame({"identifier": identifiers_, "sequence": sequences_})
    return df


def main():
    name_file = input()
    input_fasta = f"data/not_annotated/{name_file}.fasta"

    logging.basicConfig(
        level=logging.INFO,
        filename=f"data/not_annotated/have_annotation_{name_file}.log",
    )

    identifiers, sequences = collect_data(input_fasta)
    df = write_not_annotated_seqs(identifiers, sequences)
    df = filter_df(df)
    df.to_csv(f"data/embeddings/input_csv/{name_file}.csv", index=False)


if __name__ == "__main__":
    main()
