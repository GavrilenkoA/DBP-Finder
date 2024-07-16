import argparse
import pandas as pd
import requests


def get_organism_name(protein_id):
    """
    Fetches the organism name for a given protein sequence ID from the UniProt API.

    Parameters:
        protein_id (str): The ID of the protein sequence.

    Returns:
        str: The name of the organism.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        organism_name = data.get("organism", {}).get("scientificName", "Organism name not found")
        return organism_name
    else:
        return "-1"


def add_organism_column(df, protein_id_column):
    """
    Adds a column with organism names to a DataFrame based on protein IDs.

    Parameters:
        df (pd.DataFrame): The DataFrame containing protein IDs.
        protein_id_column (str): The name of the column with protein IDs.

    Returns:
        pd.DataFrame: The DataFrame with an added 'Organism' column.
    """
    df['Organism'] = df[protein_id_column].apply(get_organism_name)
    return df


def main():
    parser = argparse.ArgumentParser(description='Process protein IDs to add organism names to a CSV file.')
    parser.add_argument('input_csv_path', type=str, help='Path to the input CSV file')
    parser.add_argument('output_csv_path', type=str, help='Path to the input CSV file')

    args = parser.parse_args()

    # Load the DataFrame from the input CSV file
    df = pd.read_csv(args.input_csv_path)

    # Add the 'Organism' column to the DataFrame
    df_with_organism = add_organism_column(df, "identifier")

    # Save the updated DataFrame to the output CSV file
    df_with_organism.to_csv(args.output_csv_path, index=False)

    print(f"The DataFrame has been updated with the 'Organism' column and saved to '{args.output_csv_path}'.")


if __name__ == "__main__":
    main()
