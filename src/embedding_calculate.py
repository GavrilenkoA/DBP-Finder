import ankh
import numpy as np
import torch
import pandas as pd
import pickle
import os
from transformers import AutoTokenizer, EsmModel
import logging


def save_embeds(obj: object, data_name: str, model_name: str) -> None:
    if model_name == "esm":
        filename = f"../data/embeddings/esm_embeddings/{data_name}.pkl"
    elif model_name == "prot5":
        filename = f"../data/embeddings/prot5_embeddings/{data_name}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def calculate_embed(tokenizer, model, seq: str, model_name: str) -> np.ndarray:
    with torch.no_grad():
        if model_name == "ankh":
            inputs = tokenizer(
                [seq],
                add_special_tokens=False,
                padding=False,
                is_split_into_words=True,
                return_tensors="pt")

        elif model_name == "esm":
            inputs = tokenizer(seq, return_tensors="pt")

    inputs.to(torch.device("cuda"))
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(axis=1).view(-1).cpu().numpy()
    return embedding


def process_data(tokenizer, model, input_csv: str, model_name: str) -> None:
    def pull_data(x):
        id_ = x["identifier"]
        seq = x["sequence"]
        return id_, seq

    input_path = f"../data/embeddings/input_csv/{input_csv}"
    input_df = pd.read_csv(input_path)
    data = input_df.apply(lambda x: pull_data(x), axis=1).tolist()
    outputs = {}

    model.to(torch.device("cuda"))
    model.eval()

    for item in data:
        id_, seq = item
        embedding = calculate_embed(tokenizer, model, seq, model_name)
        outputs[id_] = embedding

        logging.info(f"{id_} does")

    data_name = os.path.splitext(input_csv)[0]
    save_embeds(outputs, data_name, model_name)


def main():
    input_csv = input()
    model_name = input()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{model_name}.log",
    )

    if model_name == "ankh":
        model, tokenizer = ankh.load_large_model()

    elif model_name == "esm":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

    process_data(tokenizer, model, input_csv, model_name)


if __name__ == "__main__":
    main()
