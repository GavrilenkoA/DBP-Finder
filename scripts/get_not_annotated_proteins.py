import logging
import time
import argparse
import yaml
import pandas as pd
import requests
from tqdm import tqdm
from utils import convert_fasta_to_df


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/annotation.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

SEED = 42


def read_yaml(yml_path: str) -> set[str]:
    with open(yml_path, "r") as file:
        data = yaml.safe_load(file)
    go_id = set(data["terms"])
    return go_id


def search_annotation(
        id_protein: str, go_id: set[str]) -> bool | None:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    page = 1
    failed_request = False
    has_annotation = False
    while True:
        # Append the page parameter to the URL for pagination
        paginated_url = f"{url}&page={page}"
        time.sleep(0.01)
        try:
            response = requests.get(
                paginated_url, timeout=15
            )  # Timeout set to 15 seconds

            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch data: {response.status_code}, id: {id_protein}"
                )
                failed_request = True
                break

            annotation_data = response.json()
            for item in annotation_data["results"]:
                if item["goId"] in go_id:
                    has_annotation = True
                    break

            if (
                annotation_data["pageInfo"]["total"] == page or annotation_data["pageInfo"]["total"] == 0
            ):
                break

            page += 1

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for id: {id_protein}")
            failed_request = True
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}, {id_protein}")
            failed_request = True
            break

    if not failed_request:
        return has_annotation


def write_not_annotated_seqs(df: pd.DataFrame, go_id: set[str]) -> dict[str, bool]:
    info = {}
    for row in tqdm(df.itertuples(), total=len(df)):
        annotation = search_annotation(row.identifier, go_id)
        if annotation is not None:
            info[row.identifier] = annotation

    return info


def main():
    # parser = argparse.ArgumentParser(description="Process FASTA file and filter based on GO terms")
    # parser.add_argument('--fasta', type=str, help='Path to FASTA file containing sequences')
    # parser.add_argument("output_yml", type=str, help='yaml file output')
    # args = parser.parse_args()

    go_id = read_yaml(yml_path="data/go_terms/GO:0003677.yml")
    df = convert_fasta_to_df("data/not_annotated/raw_fasta/merged.fasta")
    info = write_not_annotated_seqs(df, go_id)

    output_yml = "data/not_annotated/not-annotated-GO:0003677.yml"
    with open(output_yml, "w") as f:
        yaml.safe_dump(info, f)


if __name__ == "__main__":
    main()
