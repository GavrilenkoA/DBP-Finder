import argparse
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm


def get_organism_name(protein_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, timeout=10, verify=True)
        response.raise_for_status()
    except requests.exceptions.SSLError as ssl_err:
        print(f"SSL error: {ssl_err}")
        return "-1"
    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err}")
        return "-1"

    if response.status_code == 200:
        data = response.json()
        organism_name = data.get("organism", {}).get(
            "scientificName", "not found"
        )
        return organism_name
    else:
        return "not found"


def assign_kingdom(organism_name: str) -> str:
    template = {
        "Eukaryota Metazoa": "Metazoa",
        "Eukaryota Fungi": "Fungi",
        "Eukaryota Viridiplantae": "Viridiplantae",
    }

    if organism_name in template:
        return template[organism_name]
    elif organism_name.startswith("Eukaryota"):
        return "Protists"
    elif organism_name == "not found":
        return organism_name
    else:
        return organism_name.split(" ")[0]


def add_organism_column(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Processing proteins")
    df["Organism"] = df["identifier"].progress_apply(get_organism_name)
    df["Kingdom"] = df["Organism"].apply(assign_kingdom)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process protein IDs to add organism names to a CSV file."
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    # Load the DataFrame from the input CSV file
    df = pd.read_csv(args.input_csv)

    # Add the 'Organism' column to the DataFrame
    df_with_organism = add_organism_column(df)

    # Save the updated DataFrame to the output CSV file
    df_with_organism.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
