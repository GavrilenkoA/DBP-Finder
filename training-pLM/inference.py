import argparse
import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.utils import convert_fasta_to_df
from scripts.embeds import get_embeds
from .utils import (
    InferenceDataset,
    inference_ensemble_based_on_threshold,
    load_models,
    load_thresholds
)


def setup_logger():
    """Set up logger for progress visibility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_device(gpu: int) -> torch.device:
    """Set up the device for computation."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    else:
        logging.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")


def filter_df_with_warnings(df: pd.DataFrame, min_len: int = 50, max_len: int = 1024) -> pd.DataFrame:
    """Filter the dataframe and raise warnings based on criteria."""
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    original_len = len(df)

    # Filter invalid sequences
    invalid_sequences = df[~df["sequence"].apply(valid_sequence)]
    if not invalid_sequences.empty:
        logging.warning(f"{len(invalid_sequences)} sequences contain invalid characters and will be removed.")
    df = df[df["sequence"].apply(valid_sequence)]

    # Filter by sequence length
    invalid_length = df[~df["sequence"].apply(lambda x: min_len <= len(x) <= max_len)]
    if not invalid_length.empty:
        logging.warning(f"{len(invalid_length)} sequences do not meet length criteria ({min_len}-{max_len}) and will be removed.")
    df = df[df["sequence"].apply(lambda x: min_len <= len(x) <= max_len)]

    # Check for duplicate sequences
    duplicate_sequences = df.duplicated(subset=["sequence"])
    if duplicate_sequences.any():
        logging.warning(f"{duplicate_sequences.sum()} duplicate sequences found and removed.")
    df = df.drop_duplicates(subset=["sequence"])

    # Check for duplicate identifiers
    duplicate_identifiers = df.duplicated(subset=["identifier"])
    if duplicate_identifiers.any():
        logging.warning(f"{duplicate_identifiers.sum()} duplicate identifiers found and removed.")
    df = df.drop_duplicates(subset=["identifier"])

    logging.info(f"Filtered dataframe: {original_len} -> {len(df)} rows remaining.")
    return df


def process_data(df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Generate embeddings for the input dataframe."""
    logging.info("Generating embeddings...")
    outputs = get_embeds(df, model_name="ankh", device=device)
    df = pd.DataFrame({
        "identifier": list(outputs.keys()),
        "embedding": list(outputs.values())
    })
    return df


def predict(embed_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """Make predictions using pre-trained models."""
    logging.info("Loading models and thresholds...")
    models = load_models(
        prefix_name="training-pLM/checkpoints/models/DBP-Finder_",
        config_path="training-pLM/config.yml",
        device=device,
    )
    thresholds = load_thresholds(
        model_name="1_batchsize_train_p3",
        filepath="training-pLM/thresholds.json"
    )

    logging.info("Running inference...")
    inference_dataset = InferenceDataset(embed_df)
    inference_dataloader = DataLoader(
        inference_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )

    predictions_df = inference_ensemble_based_on_threshold(models, inference_dataloader, thresholds, device)
    return predictions_df


def main():
    setup_logger()
    parser = argparse.ArgumentParser(
        description="Inference DBP-Finder: Predict DBPs from FASTA file."
    )
    parser.add_argument("fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output", type=str, help="Output filename (CSV format, without extension).")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID to use (default: 0)."
    )
    args = parser.parse_args()

    # Verify input file
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"Input file '{args.fasta}' not found.")

    logging.info("Starting pipeline...")
    device = get_device(args.gpu)

    # Data processing
    logging.info("Converting FASTA to dataframe...")
    df = convert_fasta_to_df(args.fasta)
    df = filter_df_with_warnings(df)

    # Generate embeddings
    embed_df = process_data(df, device)

    # Predictions
    predictions_df = predict(embed_df, device)

    # Save outputs
    outputs_path = os.path.join("data/prediction", f"{args.output}.csv")
    predictions_df.to_csv(outputs_path, index=False)
    logging.info(f"Predictions saved to {outputs_path}")
    print(f"Predictions saved to {outputs_path}")


if __name__ == "__main__":
    main()
