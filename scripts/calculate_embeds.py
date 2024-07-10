import os
import argparse
import pandas as pd
from embeds import get_embeds


def process_data(input_csv: str, model_name: str):
    # Read CSV file
    data = pd.read_csv(input_csv)
    data_name = os.path.splitext(os.path.basename(input_csv))[0]

    # Process data using get_embeds function
    get_embeds(data, model_name, data_name)


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Process data and generate embeddings.")
    parser.add_argument("input_csv", type=str, help="Input csv")
    parser.add_argument("--model_name", type=str, default="ankh", help="Model name for embeddings (default: 'ankh')")

    args = parser.parse_args()

    # Process data
    process_data(args.input_csv, args.model_name)


if __name__ == "__main__":
    main()
