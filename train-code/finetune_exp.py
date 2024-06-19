#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ankh
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import (BatchSampler, DataLoader, Dataset,
                              SequentialSampler)


# In[2]:


import clearml
from clearml import Logger, Task


# In[3]:


clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="Finetune Ankh v2",
    output_uri=True,
)
logger = Logger.current_logger()


# In[4]:


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


# In[5]:


task.connect_configuration(config)


# In[6]:


input_dim = config["model_config"]["input_dim"]
nhead = config["model_config"]["nhead"]
hidden_dim = config["model_config"]["hidden_dim"]
num_hidden_layers = config["model_config"]["num_hidden_layers"]
num_layers = config["model_config"]["num_layers"]
kernel_size = config["model_config"]["kernel_size"]
dropout = config["model_config"]["dropout"]
pooling = config["model_config"]["pooling"]


epochs = config["training_config"]["epochs"]
lr = config["training_config"]["lr"]
factor = config["training_config"]["factor"]
patience = config["training_config"]["patience"]
min_lr = config["training_config"]["min_lr"]
batch_size = config["training_config"]["batch_size"]
seed = config["training_config"]["seed"]
num_workers = config["training_config"]["num_workers"]


# In[7]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(seed)


# In[8]:


def load_dict_from_hdf5(filename):
    """
    Load a dictionary with string keys and NumPy array values from an HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file to load the data from.

    Returns:
    dict: Dictionary with string keys and NumPy array values.
    """
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


# In[9]:


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
        "auc": auc,
        "accuracy": accuracy,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }

    return metrics


# In[10]:


DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# In[11]:


embeddings = load_dict_from_hdf5("../data/embeddings/ankh_embeddings/train_p2_2d.h5")


# In[ ]:


for key in embeddings:
    embeddings[key] = np.squeeze(embeddings[key])


# In[ ]:


embed_df = pd.DataFrame(list(embeddings.items()), columns=["identifier", "embedding"])


# In[ ]:


train_df = pd.read_csv("../data/ready_data/train_pdb2272.csv")


# In[ ]:


train_df = train_df.merge(embed_df, on="identifier")


# In[ ]:


train_df.embedding.iloc[1].shape


# In[ ]:


len(train_df.embedding.iloc[1])


# In[ ]:


gkf = GroupKFold(n_splits=5)


# In[ ]:


X = train_df["sequence"].tolist()
y = train_df["label"].tolist()
groups = train_df["cluster"].tolist()


# In[ ]:


for train_idx, test_idx in gkf.split(X, y, groups=groups):
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()
    break


# In[ ]:


train = train_df.iloc[train_idx]
test = train_df.iloc[test_idx]


# In[ ]:


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


# In[ ]:


def custom_collate_fn(batch):
    # Extract the embeddings from the batch
    embeddings = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)

    # Pad the embeddings
    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    return padded_embeddings, labels


# In[ ]:


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


# In[ ]:


train_dataset = SequenceDataset(train)
train_sampler = CustomBatchSampler(train_dataset, batch_size)
train_dataloader = DataLoader(
    train_dataset,
    num_workers=num_workers,
    batch_sampler=train_sampler,
    collate_fn=custom_collate_fn,
)


# In[ ]:


test_dataset = SequenceDataset(test)
test_sampler = CustomBatchSampler(test_dataset, batch_size)
test_dataloader = DataLoader(
    test_dataset,
    num_workers=num_workers,
    batch_sampler=test_sampler,
    collate_fn=custom_collate_fn,
)


# In[ ]:


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


# In[ ]:


binary_classification_model = binary_classification_model.to(DEVICE)


# In[ ]:


# a, b = next(iter(train_dataloader))
# output = binary_classification_model(a.to(DEVICE), b.to(DEVICE).unsqueeze(1))


# In[ ]:


optimizer = AdamW(binary_classification_model.parameters(), lr=float(lr))
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=factor, patience=patience, min_lr=float(min_lr)
)


# In[ ]:


def train(binary_classification_model, train_dataloader, optimizer):
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


# In[ ]:


def validate(binary_classification_model, test_dataloader, scheduler):
    binary_classification_model.eval()
    loss = 0.0
    all_preds = []
    all_labels = []
    logits = []

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y = y.unsqueeze(1)

            output = binary_classification_model(x, y)
            loss += output.loss.item()

            preds = (output.logits > 0.5).float()

            logits.extend(output.logits.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_loss = loss / len(test_dataloader)
    scheduler.step(epoch_loss)
    metrics = calculate_metrics(all_labels, all_preds, logits)
    return epoch_loss, metrics


# In[ ]:


for epoch in range(epochs):
    train_loss = train(binary_classification_model, train_dataloader, optimizer)
    valid_loss, metrics = validate(
        binary_classification_model, test_dataloader, scheduler
    )

    logger.report_scalar(
        title="Loss", series="train loss", value=train_loss, iteration=epoch
    )
    logger.report_scalar(
        title="Loss", series="valid loss", value=valid_loss, iteration=epoch
    )

    for metric_name, metric_value in metrics.items():
        logger.report_scalar(
            title="Metrics", series=metric_name, value=metric_value, iteration=epoch
        )

task.close()


# Testing on benchmark pdb2272

# In[ ]:





# In[ ]:




