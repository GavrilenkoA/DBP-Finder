import logging
import requests
import pandas as pd


def collect_data(fasta_file: str) -> tuple[list, list]:
    with open(fasta_file) as fi:
        lines = fi.readlines()

    identifiers = []
    sequences = []
    seq = ""

    for line in lines:
        if line.startswith(">"):
            head = line.split("|")[0].replace(">", "")
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


def get_valid_seqs(identifiers: list, sequences: list):
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    identifiers_ = []
    sequences_ = []
    for id_, seq in zip(identifiers, sequences):
        if valid_sequence(seq):
            identifiers_.append(id_)
            sequences_.append(seq)
    return identifiers_, sequences_


def check_annotation(id_protein: str) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    response = requests.get(url)
    annotation_data = response.json()

    have_annotation = False
    if annotation_data["numberOfHits"]:
        for item in annotation_data["results"]:
            if item["goId"] == "GO:0003677" or item["goId"] == "GO:0003723":
                have_annotation = True
                break

    return have_annotation


def write_not_annotated_seqs(name_file: str, identifiers: list, sequences: list) -> None:
    identifiers_ = []
    sequences_ = []

    for id_, seq in zip(identifiers, sequences):
        if not check_annotation(id_):
            identifiers_.append(id_)
            sequences_.append(seq)
        else:
            logging.info(f"{id_}")

    df = pd.DataFrame({"identifier": identifiers_, "sequence": sequences_})
    df.to_csv(f"../data/not_annotated/{name_file}.csv", index=False)


def main():
    name_file = input()
    input_fasta = f"../data/not_annotated/{name_file}.fasta"

    logging.basicConfig(
        level=logging.INFO,
        filename=f"../data/not_annotated/have_annotation_{name_file}.log",
    )

    identifiers, sequences = collect_data(input_fasta)
    identifiers, sequences = get_valid_seqs(identifiers, sequences)
    write_not_annotated_seqs(name_file, identifiers, sequences)


main()

