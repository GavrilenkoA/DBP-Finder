import torch
from torch.utils.data import BatchSampler, Dataset, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class SequenceDataset(Dataset):
    def __init__(self, df):
        self.embeds = df.embedding.tolist()
        self.labels = df.label.tolist()
        self.lengths = [len(embed) for embed in self.embeds]

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, index):
        x = self.embeds[index]
        y = self.labels[index]

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y


def custom_collate_fn(batch):
    # Extract the embeddings from the batch
    embeddings = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)

    # Pad the embeddings
    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    return padded_embeddings, labels


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        indices = list(self.sampler)
        indices.sort(
            key=lambda i: self.dataset.lengths[i], reverse=True
        )  # Sort indices by sequence length
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


def train_fn(binary_classification_model, train_dataloader, optimizer, DEVICE):
    binary_classification_model.train()
    loss = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y = y.unsqueeze(1)

        optimizer.zero_grad()
        output = binary_classification_model(x, y)
        output.loss.backward()
        optimizer.step()

        loss += output.loss.item()

    epoch_loss = loss / len(train_dataloader)
    return epoch_loss


def calculate_metrics(
    all_labels: list, all_preds: list, logits: list
) -> dict[str, float]:
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    logits = np.array(logits)

    auc = roc_auc_score(all_labels, logits)
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    f1 = f1_score(all_labels, all_preds)

    metrics = {
        "Accuracy": accuracy,
        "Sensitivity": recall,
        "Specificity": specificity,
        "Precision": precision,
        "AUC": auc,
        "F1": f1,
        "MCC": mcc,
    }

    return metrics


def validate_fn(binary_classification_model, valid_dataloader, DEVICE):
    binary_classification_model.eval()
    loss = 0.0
    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y = y.unsqueeze(1)

            output = binary_classification_model(x, y)
            loss += output.loss.item()

    epoch_loss = loss / len(valid_dataloader)
    return epoch_loss