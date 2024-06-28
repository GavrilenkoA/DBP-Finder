import requests
import time
import pandas as pd
from tqdm import tqdm
import logging
from utils import convert_fasta_to_df, filter_df

# rna binding "GO:0003723 dna binding "GO:0003677 Binding to a nucleic acid GO:0003676"

organism = input()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"logs/{organism}_annot.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


def search_annotation(
    id_protein: str, go_id: set[str] = {"GO:0003723", "GO:0003677", "GO:0003676"}
) -> bool:
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
                has_annotation
                or page == annotation_data["pageInfo"]["total"]
                or annotation_data["pageInfo"]["total"] == 0
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


def write_not_annotated_seqs(df: pd.DataFrame) -> pd.DataFrame:
    identifiers = []
    sequences = []

    for row in tqdm(df.itertuples(), total=len(df)):
        if not search_annotation(row.identifier):
            identifiers.append(row.identifier)
            sequences.append(row.sequence)
        else:
            logger.error(f"{row.identifier} has annotation")

    df = pd.DataFrame({"identifier": identifiers, "sequence": sequences})
    return df


def main():
    input_fasta = f"data/not_annotated/{organism}.fasta"
    df = convert_fasta_to_df(input_fasta)
    df = filter_df(df)
    df = write_not_annotated_seqs(df)
    df.to_csv(f"data/not_annotated/{organism}.csv", index=False)


if __name__ == "__main__":
    main()
