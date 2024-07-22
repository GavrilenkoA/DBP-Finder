import argparse
import os

import pandas as pd
import torch
from embeds import get_embeds


def process_data(input_csv: str, model_name: str, device: torch.device):
    # Read CSV file
    data = pd.read_csv(input_csv)
    data_name = os.path.splitext(os.path.basename(input_csv))[0]

    # Process data using get_embeds function
    get_embeds(data, model_name, data_name, device)


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process data and generate embeddings."
    )
    parser.add_argument("input_csv", type=str, help="Input csv")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ankh",
        help="Model name for embeddings (default: 'ankh')",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:3", help="Device to use (default: 'cuda:1')"
    )
    args = parser.parse_args()

    # Determine the device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Process data
    process_data(args.input_csv, args.model_name, device)


if __name__ == "__main__":
    main()
