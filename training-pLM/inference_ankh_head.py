import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_prepare import get_embed_clustered_df
from utils import (InferenceDataset, inference_ensemble_based_on_threshold,
                   load_models, load_thresholds)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = "pdb20000"


def predict(test_data):
    models = load_models(prefix_name=f"checkpoints/{test_data}_", config_path="config.yml")
    thresholds = load_thresholds(model_name=f"1_batchsize_{test_data}")
    test_df = get_embed_clustered_df(
        embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{test_data}_2d.h5",
        csv_path=f"../data/embeddings/input_csv/{test_data}.csv",
    )

    inference_dataset = InferenceDataset(test_df)
    inference_dataloader = DataLoader(
        inference_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )

    predictions_df = inference_ensemble_based_on_threshold(models, inference_dataloader, thresholds, DEVICE)
    test_df = test_df[["identifier", "sequence", "label"]]
    taxon_df = pd.read_csv(f"../data/taxon/{test_data}.csv")
    test_df = test_df.merge(taxon_df, on="identifier")
    predictions_df = predictions_df.merge(test_df, on="identifier")
    predictions_df.to_csv(f"../data/prediction/{test_data}.csv", index=False)


predict(test_data)
