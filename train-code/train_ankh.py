# %%
import ankh
import numpy as np
import pandas as pd
import torch
import yaml
import logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_prepare import make_folds, prepare_embed_df
from torch_utils import (
    SequenceDataset,
    custom_collate_fn,
    CustomBatchSampler,
    train_fn,
    validate_fn,
    evaluate_fn,
)

# %%
import clearml
from clearml import Logger, Task

# %%
clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="Finetune Ankh pdb 1000",
    output_uri=True,
)
logger = Logger.current_logger()

# %%
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# %%
task.connect_configuration(config)

# %%
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

# %%
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# %%
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(seed)

# %% [markdown]
# train - pd dataframe

# %%
# binary_classification_model = ankh.ConvBertForBinaryClassification(
#     input_dim=1536,
#     nhead=7,
#     hidden_dim=1723,
#     num_hidden_layers=2,
#     num_layers=1,
#     kernel_size=7,
#     dropout=dropout,
#     pooling=pooling,
# )

# %%
# binary_classification_model = binary_classification_model.to(DEVICE)

# %%
# a, b = next(iter(train_dataloader))

# %%
# output = binary_classification_model(a.to(DEVICE), b.to(DEVICE).unsqueeze(1))

# %%
# optimizer = AdamW(binary_classification_model.parameters(), lr=float(lr))
# scheduler = ReduceLROnPlateau(
#     optimizer, mode="min", factor=factor, patience=patience, min_lr=float(min_lr)
# )

# %%
df = prepare_embed_df(csv_path="../data/ready_data/train_pdb1000.csv")
train_folds, valid_folds = make_folds(df)

# %%
df.label.value_counts()

# %%
for i in range(len(train_folds)):
    train_dataset = SequenceDataset(train_folds[i])
    train_sampler = CustomBatchSampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_sampler=train_sampler,
        collate_fn=custom_collate_fn,
    )

    valid_dataset = SequenceDataset(valid_folds[i])
    valid_sampler = CustomBatchSampler(valid_dataset, batch_size)
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=num_workers,
        batch_sampler=valid_sampler,
        collate_fn=custom_collate_fn,
    )

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

    binary_classification_model = binary_classification_model.to(DEVICE)
    optimizer = AdamW(binary_classification_model.parameters(), lr=float(lr))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=float(min_lr)
    )

    best_val_loss = float("inf")
    best_model_path = f"checkpoints/pdb1000_best_model_{i}.pth"

    for epoch in range(epochs):
        train_loss = train_fn(
            binary_classification_model, train_dataloader, optimizer, DEVICE
        )
        valid_loss, metrics = validate_fn(
            binary_classification_model, valid_dataloader, scheduler, DEVICE
        )

        logger.report_scalar(
            title=f"Loss model {i}",
            series="train loss",
            value=train_loss,
            iteration=epoch,
        )
        logger.report_scalar(
            title=f"Loss model {i}",
            series="valid loss",
            value=valid_loss,
            iteration=epoch,
        )

        for metric_name, metric_value in metrics.items():
            logger.report_scalar(
                title=f"Metrics model {i}",
                series=metric_name,
                value=metric_value,
                iteration=epoch,
            )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(binary_classification_model.state_dict(), best_model_path)
            message = f"Saved Best Model on epoch {epoch} with Validation Loss: {best_val_loss}"
            logger.report_text(message, level=logging.DEBUG, print_console=False)

# %% [markdown]
# Inference average best models

# %%
model_0 = ankh.ConvBertForBinaryClassification(
    input_dim=input_dim,
    nhead=nhead,
    hidden_dim=hidden_dim,
    num_hidden_layers=num_hidden_layers,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout=dropout,
    pooling=pooling,
)

model_1 = ankh.ConvBertForBinaryClassification(
    input_dim=input_dim,
    nhead=nhead,
    hidden_dim=hidden_dim,
    num_hidden_layers=num_hidden_layers,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout=dropout,
    pooling=pooling,
)

model_2 = ankh.ConvBertForBinaryClassification(
    input_dim=input_dim,
    nhead=nhead,
    hidden_dim=hidden_dim,
    num_hidden_layers=num_hidden_layers,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout=dropout,
    pooling=pooling,
)

model_3 = ankh.ConvBertForBinaryClassification(
    input_dim=input_dim,
    nhead=nhead,
    hidden_dim=hidden_dim,
    num_hidden_layers=num_hidden_layers,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout=dropout,
    pooling=pooling,
)

model_4 = ankh.ConvBertForBinaryClassification(
    input_dim=input_dim,
    nhead=nhead,
    hidden_dim=hidden_dim,
    num_hidden_layers=num_hidden_layers,
    num_layers=num_layers,
    kernel_size=kernel_size,
    dropout=dropout,
    pooling=pooling,
)

# %%
model_0.load_state_dict(torch.load("checkpoints/pdb1000_best_model_0.pth"))
model_1.load_state_dict(torch.load("checkpoints/pdb1000_best_model_1.pth"))
model_2.load_state_dict(torch.load("checkpoints/pdb1000_best_model_2.pth"))
model_3.load_state_dict(torch.load("checkpoints/pdb1000_best_model_3.pth"))
model_4.load_state_dict(torch.load("checkpoints/pdb1000_best_model_4.pth"))

# %%
models = [model_0, model_1, model_2, model_3, model_4]

# %% [markdown]
# Testing on benchmark pdb2272

# %%
test_df = prepare_embed_df(
    embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/pdb1000_2d.h5",
    csv_path="../data/embeddings/input_csv/pdb1000.csv",
)

# %%
test_df.label.value_counts()

# %%
test_df

# %%
testing_set = SequenceDataset(test_df)
testing_dataloader = DataLoader(
    testing_set,
    num_workers=num_workers,
    shuffle=False,
    batch_size=1,
)

# %%
metrics = evaluate_fn(models, testing_dataloader, DEVICE)

# %%
metrics_df = pd.DataFrame(metrics, index=["pdb1000"])

# %%
logger.report_table(title="pdb1000", series="Metrics", table_plot=metrics_df)

# %%
task.close()

# %%
