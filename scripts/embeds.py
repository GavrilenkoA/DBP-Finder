import os

import ankh
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_model_tokenizer(model_name: str):
    if model_name == "ankh":
        model, tokenizer = ankh.load_large_model()

    elif model_name == "esm":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

    elif model_name == "prot5":
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
        )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    return model, tokenizer


def calculate_embeds(
    tokenizer, model, seq: str, model_name: str, device: torch.device
) -> np.ndarray:
    if model_name == "ankh":
        inputs = tokenizer(
            [seq],
            add_special_tokens=False,
            padding=False,
            is_split_into_words=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs.to(device)
            output = model(**inputs)

    elif model_name == "esm":
        inputs = tokenizer(seq, return_tensors="pt")

        with torch.no_grad():
            inputs.to(device)
            output = model(**inputs)

    elif model_name == "prot5":
        item = []
        for i in range(len(seq)):
            if i != 0 and i != len(seq):
                item.append(" ")
            item.append(seq[i])

        item = ["".join(item)]

        ids = tokenizer.batch_encode_plus(item, add_special_tokens=False, padding=False)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = output.last_hidden_state.cpu().numpy()  # mean(axis=1).view(-1)
    embedding = np.squeeze(embedding)
    return embedding


def get_embeds(
    df: pd.DataFrame, model_name: str,
        device: torch.device) -> dict[str, np.ndarray]:

    model, tokenizer = select_model_tokenizer(model_name)
    model.to(device)
    model.eval()

    identifiers = df["identifier"].tolist()
    sequences = df["sequence"].tolist()

    outputs = {}
    for i in tqdm(range(len(df)), total=len(df), desc="Embeddings calculation"):
        identifier = identifiers[i]
        seq = sequences[i]

        embedding = calculate_embeds(tokenizer, model, seq, model_name, device)
        outputs[identifier] = embedding

    return outputs
