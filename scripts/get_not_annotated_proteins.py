import requests
import pandas as pd
from tqdm import tqdm
from utils import filter_df, extract_columns

# rna binding "GO:0003723 dna binding "GO:0003677"


def have_annotation(id_protein: str, go_id: set[str] = {"GO:0003723"}) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    page = 1
    has_annotation = False

    while True:
        # Append the page parameter to the URL for pagination
        paginated_url = f"{url}&page={page}"
        response = requests.get(paginated_url)

        # Check for errors in response
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}, url:{paginated_url}")
            break

        annotation_data = response.json()
        # Iterate through results to find the GO term
        for item in annotation_data["results"]:
            if item["goId"] in go_id:
                has_annotation = True
                break

        if has_annotation or page == annotation_data["pageInfo"]["total"]:
            break

        page += 1  # Increment the page number to fetch the next page

    return has_annotation


def write_not_annotated_seqs(identifiers: list, sequences: list) -> pd.DataFrame:
    identifiers_ = []
    sequences_ = []
    labels_ = []

    for id_, seq in tqdm(zip(identifiers, sequences), total=len(identifiers)):
        identifiers_.append(id_)
        sequences_.append(seq)

        if not have_annotation(id_):
            labels_.append(0)
        else:
            labels_.append(1)

    df = pd.DataFrame(
        {"identifier": identifiers_, "sequence": sequences_, "label": labels_}
    )
    return df


def main():
    # name_file = "go_0003677_swissprot"
    # input_fasta = f"data/uniprot/{name_file}.fasta"

    # identifiers, sequences = extract_columns(input_fasta)
    df = pd.read_csv("data/embeddings/input_csv/pdb20000.csv")
    identifiers = df["identifier"].to_list()
    sequences = df["sequence"].to_list()

    df = write_not_annotated_seqs(identifiers, sequences)
    # df = filter_df(df)
    df.to_csv("data/processed/rna_binders_pdb20000.csv", index=False)


if __name__ == "__main__":
    main()
