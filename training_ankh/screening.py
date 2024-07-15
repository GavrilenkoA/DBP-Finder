import torch
from data_prepare import load_embeddings_to_df
from torch.utils.data import DataLoader
from torch_utils import InferenceDataset, inference, load_models


def main():
    input_data = "../data/embeddings/ankh_embeddings/not_annotated_seqs_v1_2d.h5"
    df = load_embeddings_to_df(input_data)
    models = load_models()
    inference_dataset = InferenceDataset(df)
    inference_dataloader = DataLoader(
        inference_dataset,
        shuffle=False,
        batch_size=1,
    )
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions_df = inference(models, inference_dataloader, DEVICE)
    predictions_df.to_csv("../data/not_annotated/predictions.csv", index=False)


if __name__ == "__main__":
    main()
