import logging
import time
import argparse
import yaml
import pandas as pd
import requests
from tqdm import tqdm
from utils import convert_fasta_to_df, filter_df

# rna binding "GO:0003723 dna binding "GO:0003677 Binding to a nucleic acid GO:0003676"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/model_org.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


def read_yaml(yml_path: str) -> set[str]:
    with open(yml_path, "r") as file:
        data = yaml.safe_load(file)
    go_id = set(data["na_terms"])
    return go_id


def search_annotation(
        id_protein: str, go_id: set[str]) -> bool:
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={id_protein}"
    page = 1
    has_annotation = False

    while True:
        # Append the page parameter to the URL for pagination
        paginated_url = f"{url}&page={page}"
        time.sleep(0.01)
        try:
            response = requests.get(
                paginated_url, timeout=15
            )  # Timeout set to 15 seconds

            # Check for errors in response
            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch data: {response.status_code}, id: {id_protein}"
                )
                break

            annotation_data = response.json()
            # Iterate through results to find the GO term
            for item in annotation_data["results"]:
                if item["goId"] in go_id:
                    has_annotation = True
                    break

            if (
                has_annotation or page == annotation_data["pageInfo"]["total"] or annotation_data["pageInfo"]["total"] == 0
            ):
                break

            page += 1  # Increment the page number to fetch the next page

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for id: {id_protein}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}, {id_protein}")
            break

    return has_annotation


def write_not_annotated_seqs(df: pd.DataFrame, go_id: set[str]) -> dict[str, bool]:
    info = {}
    for row in tqdm(df.itertuples(), total=len(df)):
        annotation = search_annotation(row.identifier, go_id)
        info[row.identifier] = annotation
    return info


def main():
    parser = argparse.ArgumentParser(description="Process FASTA file and filter based on GO terms")
    parser.add_argument('--fasta', type=str, help='Path to FASTA file containing sequences')
    parser.add_argument('--output_yml', type=str, help='yaml file output')

    args = parser.parse_args()
    yml_path = "data/go_terms/na_terms.yml"
    go_id = read_yaml(yml_path)

    df = convert_fasta_to_df(args.fasta)
    df = filter_df(df)
    info = write_not_annotated_seqs(df, go_id)

    with open(args.output_yml, 'w') as f:
        yaml.safe_dump(info, f)


if __name__ == "__main__":
    main()
