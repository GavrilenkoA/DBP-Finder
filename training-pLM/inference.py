import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..scripts.utils import convert_fasta_to_df, filter_df
from ..scripts.embeds import get_embeds
from .utils import InferenceDataset, inference_ensemble_based_on_threshold, load_models, load_thresholds


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_data(df: pd.DataFrame) -> None:
    outputs = get_embeds(df, model_name="ankh", device=DEVICE)
    df = pd.DataFrame({
        "identifier": list(outputs.keys()),
        "embedding": list(outputs.values())})

    return df


def predict(embed_df) -> pd.DataFrame:
    models = load_models(prefix_name="checkpoints/models/DBP-Finder_", config_path="config.yml")
    thresholds = load_thresholds(model_name="1_batchsize_train_p3")

    inference_dataset = InferenceDataset(embed_df)
    inference_dataloader = DataLoader(
        inference_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )
    predictions_df = inference_ensemble_based_on_threshold(models, inference_dataloader, thresholds, DEVICE)
    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Inference DBP-Finder"
    )
    parser.add_argument("fasta_path", type=str, help="Input fasta")
    args = parser.parse_args()

    df = convert_fasta_to_df(args.fasta_path)
    df = filter_df(df)

    embed_df = process_data(df)
    predictions_df = predict(embed_df)

    predictions_df.to_csv("data/prediction/predictions.csv", index=False)
