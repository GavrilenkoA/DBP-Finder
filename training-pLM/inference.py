import argparse
import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.utils import convert_fasta_to_df, chunk_dataframe, hash_suffix, postprocess
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

    # Check for duplicate sequences
    duplicate_sequences = df.duplicated(subset=["sequence"])
    if duplicate_sequences.any():
        logging.warning(f"{duplicate_sequences.sum()} duplicate sequences found and removed.")
        df = df.drop_duplicates(subset=["sequence"])

    # Check for duplicate identifiers
    duplicate_identifiers = df.duplicated(subset=["identifier"])
    if duplicate_identifiers.any():
        logging.warning(f"{duplicate_identifiers.sum()} duplicate identifiers found, adding suffix.")
        df["identifier"] = df.groupby("identifier")["identifier"].transform(
            lambda x: x if len(x) == 1 else x + "_" + x.apply(lambda y: hash_suffix(y)))

    # Filter invalid sequences
    invalid_sequences = df[~df["sequence"].apply(valid_sequence)]
    if not invalid_sequences.empty:
        logging.warning(f"{len(invalid_sequences)} sequences contain none-canonical amino acids.")

    # Filter by sequence length
    short_sequences = df[df["sequence"].apply(lambda x: len(x) < min_len)]
    long_sequences = df[df["sequence"].apply(lambda x: len(x) > max_len)]
    if not short_sequences.empty:
        logging.warning(f"{len(short_sequences)} sequences are too short, have length less than {min_len}")
    if not long_sequences.empty:
        logging.warning(f"{len(long_sequences)} sequences are too long, have length greater than {max_len}")
        df = df[~df["sequence"].apply(lambda x: len(x) > max_len)]
        long_sequences_chunked = chunk_dataframe(long_sequences, chunk_size=1024, overlap=512)
        df = pd.concat([df, long_sequences_chunked])
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
    predictions_df = postprocess(predictions_df)

    # Save outputs
    outputs_path = os.path.join("data/prediction", f"{args.output}.csv")
    predictions_df.to_csv(outputs_path, index=False)
    logging.info(f"Predictions saved to {outputs_path}")
    print(f"Predictions saved to {outputs_path}")


if __name__ == "__main__":
    main()
