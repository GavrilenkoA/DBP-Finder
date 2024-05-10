import ankh
import numpy as np
import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer


def select_model_tokenizer(model_name: str) -> None:
    if model_name == "ankh":
        model, tokenizer = ankh.load_large_model()

    elif model_name == "esm":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

    elif model_name == "prot5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    return model, tokenizer


def save_embeds(obj, data_name: str, model_name: str) -> None:
    filename = f"data/embeddings/{model_name}_embeddings/{data_name}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def calculate_embeds(tokenizer, model, seq: str, model_name: str) -> np.ndarray:
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

        ids = tokenizer.batch_encode_plus(item, add_special_tokens=False,
                                          padding=False)
        input_ids = torch.tensor(ids['input_ids']).to(torch.device("cuda"))
        attention_mask = torch.tensor(ids['attention_mask']).to(torch.device("cuda"))

        with torch.no_grad():
            embedding = model(input_ids=input_ids,
                              attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.mean(axis=1).view(-1).cpu().numpy()

    return embedding


def get_embeds(input_df: pd.DataFrame, data_name: str, model_name: str = "ankh") -> None:
    def pull_data(x):
        id_ = x["identifier"]
        seq = x["sequence"]
        return id_, seq

    data = input_df.apply(lambda x: pull_data(x), axis=1).tolist()
    outputs = {}

    model, tokenizer = select_model_tokenizer(model_name)
    model.to(torch.device("cuda"))
    model.eval()

    for item in data:
        id_, seq = item
        embedding = calculate_embeds(tokenizer, model, seq, model_name)
        outputs[id_] = embedding

    save_embeds(outputs, data_name, model_name)
