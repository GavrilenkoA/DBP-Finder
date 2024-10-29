import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))  # Directly create a list of indices

    def __iter__(self):
        # Sort indices by sequence length in the dataset
        sorted_indices = sorted(
            self.indices, key=lambda i: self.dataset.lengths[i], reverse=True
        )

        # Create batches of indices
        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        # Shuffle batches if required
        if self.shuffle:
            random.shuffle(batches)

        # Yield batches
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch, tokenizer, labels_flag=True):
    sequences = [item[0] for item in batch]
    identifiers = None

    if labels_flag:
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)

    if len(batch[0]) == 3 and labels_flag:
        identifiers = [item[2] for item in batch]
    elif len(batch[0]) == 2 and not labels_flag:
        identifiers = [item[1] for item in batch]

    # Tokenize the sequences
    tokenized = tokenizer(
        sequences,
        padding="longest",
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    output = {"input_ids": input_ids, "attention_mask": attention_mask}

    if labels_flag:
        output["labels"] = labels

    if identifiers is not None:
        output["identifiers"] = identifiers

    return output


class SequenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].tolist()
        self.lengths = [len(sequence) for sequence in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = self.sequences[index]
        y = self.labels[index]

        y = torch.tensor(y, dtype=torch.float)
        return x, y


class SequenceDatasetWithID(SequenceDataset):
    def __init__(self, df):
        super().__init__(df)
        self.identifier = df["identifier"].tolist()

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        id_value = self.identifier[index]
        return x, y, id_value


class InferenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = df["sequence"].tolist()
        self.identifier = df["identifier"].tolist()
        self.lengths = [len(sequence) for sequence in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = self.sequences[index]
        id_value = self.identifier[index]
        return x, id_value


def dataloader_prepare(
    data: pd.DataFrame,
    tokenizer,
    dataset_class,
    batch_size=1,
    shuffle=False,
    labels_flag=True,
):
    dataset = dataset_class(data)

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=lambda x: collate_fn(x, tokenizer, labels_flag=labels_flag),
    )

    return dataloader
