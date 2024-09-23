import torch
from torch.utils.data import Dataset, Sampler
import random


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))  # Directly create a list of indices

    def __iter__(self):
        # Sort indices by sequence length in the dataset
        sorted_indices = sorted(self.indices, key=lambda i: self.dataset.lengths[i], reverse=True)

        # Create batches of indices
        batches = [
            sorted_indices[i:i + self.batch_size]
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


def collate_fn(batch, tokenizer):
    # Extract the sequences and labels from the batch
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float16)

    # Tokenize the sequences
    tokenized = tokenizer(
        sequences,
        padding="longest",
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True)

    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels}


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


# def test_prepare(num_workers, tokenizer, batch_size=1):
#     test = pd.read_csv("../data/embeddings/input_csv/pdb2272.csv")
#     dataloader = dataset_prepare(
#         test, tokenizer, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )
#     return dataloader
