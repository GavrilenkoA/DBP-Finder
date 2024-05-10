import logging
import requests
import pandas as pd
from tqdm import tqdm
from utils import filter_df, extract_columns


def have_annotation(id_protein: str, go_id: set[str] = {"GO:0003677", "GO:0003723"}) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    page = 1
    has_annotation = False

    while True:
        # Append the page parameter to the URL for pagination
        paginated_url = f"{url}&page={page}"
        response = requests.get(paginated_url)
        annotation_data = response.json()

        # Check for errors in response
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        # Iterate through results to find the GO term
        for item in annotation_data["results"]:
            if item["goId"] in go_id:  # rna binding "GO:0003723"
                has_annotation = True
                break

        if has_annotation or page == annotation_data['pageInfo']['total']:
            break

        page += 1  # Increment the page number to fetch the next page

    return has_annotation


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
    df = pd.read_csv("data/embeddings/input_csv/train_p2.csv")
    neg_samples = df.loc[df["label"] == 0]

    # input_fasta = f"data/uniprot/{name_file}.fasta"

    logging.basicConfig(
        level=logging.INFO,
        filename="logs/have_annotation_train_neg_samples.log")

    # identifiers, sequences = extract_columns(input_fasta)
    identifiers = neg_samples.loc[:, "identifier"].to_list()
    sequences = neg_samples.loc[:, "sequence"].to_list()

    df = write_not_annotated_seqs(identifiers, sequences)
    # df = filter_df(df)
    # df.to_csv(f"data/embeddings/input_csv/{name_file}.csv", index=False)


if __name__ == "__main__":
    main()
