import ankh
import numpy as np
import torch
import pandas as pd
import pickle
import os
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer
import logging


def save_embeds(obj: object, data_name: str, model_name: str) -> None:
    if model_name == "esm":
        filename = f"../data/embeddings/esm_embeddings/{data_name}.pkl"
    elif model_name == "prot5":
        filename = f"../data/embeddings/prot5_embeddings/{data_name}.pkl"
    elif model_name == "ankh":
        filename = f"../data/embeddings/ankh_embeddings/{data_name}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def calculate_embed(tokenizer, model, seq: str, model_name: str) -> np.ndarray:
    if model_name == "ankh":
        inputs = tokenizer(
            [seq],
            add_special_tokens=False,
            padding=False,
            is_split_into_words=True,
            return_tensors="pt")

        with torch.no_grad():
            inputs.to(torch.device("cuda"))
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(axis=1).view(-1).cpu().numpy()

    elif model_name == "esm":
        inputs = tokenizer(seq, return_tensors="pt")

        with torch.no_grad():
            inputs.to(torch.device("cuda"))
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(axis=1).view(-1).cpu().numpy()

    elif model_name == "prot5":
        item = []
        for i in range(len(seq)):
            if i != 0 and i != len(seq):
                item.append(" ")
            item.append(seq[i])

        item = ["".join(item)]

        ids = tokenizer.batch_encode_plus(item, add_special_tokens=False, padding=False)
        input_ids = torch.tensor(ids['input_ids']).to(torch.device("cuda"))
        attention_mask = torch.tensor(ids['attention_mask']).to(torch.device("cuda"))

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.mean(axis=1).view(-1).cpu().numpy()

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
        filename=f"{input_csv}_{model_name}.log",
    )

    if model_name == "ankh":
        model, tokenizer = ankh.load_large_model()

    elif model_name == "esm":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

    elif model_name == "prot5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    process_data(tokenizer, model, input_csv, model_name)


if __name__ == "__main__":
    main()
