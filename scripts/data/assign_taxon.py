import argparse

import pandas as pd
import requests
from tqdm import tqdm


def assign_kingdom(taxon: str) -> str:
    template = {
        "Eukaryota Metazoa": "Metazoa",
        "Eukaryota Fungi": "Fungi",
        "Eukaryota Viridiplantae": "Viridiplantae",
    }

    if taxon in template:
        return template[taxon]
    elif taxon.startswith("Eukaryota"):
        return "Protists"
    elif taxon == "not found":
        return taxon
    else:
        return taxon.split(" ")[0]


def add_taxon(protein_id: str) -> tuple[str, str] | None:
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return "not found"

    info = response.json().get("organism", {})
    if not info:
        return None

    organism_name = info.get("scientificName", "not found")
    lineage = info.get("lineage", "not found")
    taxon = " ".join(lineage[:2])
    kingdom = assign_kingdom(taxon)
    return organism_name, kingdom


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Processing proteins")
    df[['organism', 'kingdom']] = df['identifier'].progress_apply(lambda x: pd.Series(add_taxon(x)))
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
    df_with_organism = process_columns(df)

    # Save the updated DataFrame to the output CSV file
    df_with_organism.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
