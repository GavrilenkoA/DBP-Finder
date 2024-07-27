import logging

import ankh
import clearml
import numpy as np
import torch
import yaml
from clearml import Logger, Task
from data_prepare import get_embed_clustered_df, make_folds
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_utils import (CustomBatchSampler, SequenceDataset,
                         custom_collate_fn, train_fn, validate_fn)

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

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


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(seed)


df = get_embed_clustered_df(
    embedding_path="../data/embeddings/ankh_embeddings/train_p3_2d.h5",
    csv_path="../data/splits/train_pdb1000.csv",
)
train_folds, valid_folds = make_folds(df)


clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="train_pdb1000",
    output_uri=True,
)
logger = Logger.current_logger()
task.connect_configuration(config)

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

task.close()
