import argparse
import os

import pandas as pd
import torch
from embeds import get_embeds


def process_data(input_csv: str, model_name: str, device: torch.device, output_prefix: str):
    data = pd.read_csv(input_csv)
    data_name = os.path.splitext(os.path.basename(input_csv))[0]

    # Process data using get_embeds function
    get_embeds(data, model_name, data_name, device, output_prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Process data and generate embeddings."
    )
    parser.add_argument("input_csv", type=str, help="Input csv")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ankh",
        help="Model name for calculating embeddings",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="Device to use"
    )
    parser.add_argument(
        "--output_prefix", type=str, default="../../../ssd2/dbp_finder/ankh_embeddings", help="Directory to save embeddings"
    )
    args = parser.parse_args()

    # Determine the device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Process data
    process_data(args.input_csv, args.model_name, device, args.output_prefix)


if __name__ == "__main__":
    main()
