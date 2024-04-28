import logging
import requests
import pandas as pd
from tqdm import tqdm
from utils import filter_df, extract_columns, convert_fasta_to_df


def have_annotation(id_protein: str) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/\
            search?geneProductId={id_protein}"
    response = requests.get(url)
    annotation_data = response.json()

    annotation = False
    if annotation_data["numberOfHits"]:
        for item in annotation_data["results"]:
            if item["goId"] in {"GO:0003677", "GO:0003723"}:
                annotation = True
                break
    return annotation


def write_not_annotated_seqs(identifiers: list,
                             sequences: list) -> pd.DataFrame:
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

    identifiers, sequences = extract_columns(column1="identifier", column2="sequence")(convert_fasta_to_df(input_fasta))

    df = write_not_annotated_seqs(identifiers, sequences)
    df = filter_df(df)
    df.to_csv(f"data/embeddings/input_csv/{name_file}.csv", index=False)


if __name__ == "__main__":
    main()
