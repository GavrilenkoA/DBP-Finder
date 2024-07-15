import ankh
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, Dataset, SequentialSampler


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


class InferenceDataset(Dataset):
    def __init__(self, df):
        self.identifiers = df.identifier.tolist()
        self.embeds = df.embedding.tolist()

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        id_ = self.identifiers[index]

        x = self.embeds[index]
        x = torch.tensor(x, dtype=torch.float)
        return x, id_


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


def validate_fn(binary_classification_model, valid_dataloader, scheduler, DEVICE):
    binary_classification_model.eval()
    loss = 0.0
    all_preds = []
    all_labels = []
    logits = []

    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y = y.unsqueeze(1)

            output = binary_classification_model(x, y)
            loss += output.loss.item()

            preds = (output.logits > 0.5).float()

            logits.extend(output.logits.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_loss = loss / len(valid_dataloader)
    scheduler.step(epoch_loss)
    metrics = calculate_metrics(all_labels, all_preds, logits)
    return epoch_loss, metrics


def collect_logits_labels(models, testing_dataloader, DEVICE):
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in testing_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y = y.unsqueeze(1)
            ens_logits = []

            for model in models:
                model.eval()
                model = model.to(DEVICE)
                output = model(x, y)

                logits = output.logits
                ens_logits.append(logits)

            ens_logits = torch.stack(ens_logits, dim=0)
            ens_logits = torch.mean(ens_logits, dim=0)

            all_logits.extend(ens_logits.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return all_logits, all_labels


def inference(models, testing_dataloader, DEVICE) -> pd.DataFrame:
    identifiers = []
    scores = []

    with torch.no_grad():
        for id_, x in testing_dataloader:
            x = x.to(DEVICE)
            ens_logits = []

            for model in models:
                model.eval()
                model = model.to(DEVICE)
                output = model(x)

                logits = output.logits
                ens_logits.append(logits)

            ens_logits = torch.stack(ens_logits, dim=0)
            ens_logits = torch.mean(ens_logits, dim=0)
            prob_score = torch.sigmoid(ens_logits)

            identifiers.extend(id_)
            scores.append(prob_score.cpu().numpy().item())

    df = pd.DataFrame({"identifier": identifiers, "score": scores})
    return df


def evaluate_fn(all_labels: list, all_logits: list) -> dict[str, float]:
    all_preds = [1 if logit > 0.5 else 0 for logit in all_logits]
    metrics = calculate_metrics(all_labels, all_preds, all_logits)
    return metrics


def load_models(
    prefix_name: str = "checkpoints/DBP-finder_",
    num_models: int = 5,
    config_path: str = "config.yml",
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_dim = config["model_config"]["input_dim"]
    nhead = config["model_config"]["nhead"]
    hidden_dim = config["model_config"]["hidden_dim"]
    num_hidden_layers = config["model_config"]["num_hidden_layers"]
    num_layers = config["model_config"]["num_layers"]
    kernel_size = config["model_config"]["kernel_size"]
    dropout = config["model_config"]["dropout"]
    pooling = config["model_config"]["pooling"]

    models = []
    for i in range(num_models):
        binary_classification_model = ankh.ConvBertForBinaryClassification(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )

        path_model = prefix_name + f"{i}.pth"
        binary_classification_model.load_state_dict(torch.load(path_model))
        binary_classification_model.eval()  # Set the model to evaluation mode
        models.append(binary_classification_model)

    return models
