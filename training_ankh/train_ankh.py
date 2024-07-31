import logging

import ankh
import clearml
import numpy as np
import pandas as pd
import torch
import yaml
from clearml import Logger, Task
from data_prepare import get_embed_clustered_df, make_folds
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_utils import (
    CustomBatchSampler,
    SequenceDataset,
    custom_collate_fn,
    evaluate_fn,
    train_fn,
    validate_fn,
)

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


DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(seed)
input_data = input()


df = get_embed_clustered_df(
    embedding_path="../data/embeddings/ankh_embeddings/train_p3_2d.h5",
    csv_path=f"../data/splits/train_{input_data}.csv",
)
train_folds, valid_folds = make_folds(df)


clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name=f"train_{input_data}",
    output_uri=True,
)
logger = Logger.current_logger()
task.connect_configuration(config)

models_data = {}

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

    model = ankh.ConvBertForBinaryClassification(
        input_dim=input_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        pooling=pooling,
    )

    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=float(lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_fn(model, train_dataloader, optimizer, DEVICE)
        scheduler.step()
        valid_loss, metrics_dict, optimal_threshold = validate_fn(
            model, valid_dataloader, DEVICE
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

        for metric_name, metric_value in metrics_dict.items():
            logger.report_scalar(
                title=f"Metrics model {i}",
                series=metric_name,
                value=metric_value,
                iteration=epoch,
            )

        if valid_loss < best_val_loss:
            message = (
                f"Model {i} on epoch {epoch} with Validation Loss: {best_val_loss}"
            )
            logger.report_text(message, level=logging.DEBUG, print_console=False)

            message = f"Model {i} on epoch {epoch} with optimal threshold: {optimal_threshold}"
            logger.report_text(message, level=logging.DEBUG, print_console=False)

            models_data[i] = (model, optimal_threshold)
            best_val_loss = valid_loss


test_df = get_embed_clustered_df(
    embedding_path=f"../data/embeddings/ankh_embeddings/{input_data}_2d.h5",
    csv_path=f"../data/embeddings/input_csv/{input_data}.csv",
)

testing_set = SequenceDataset(test_df)
testing_dataloader = DataLoader(
    testing_set,
    num_workers=num_workers,
    shuffle=False,
    batch_size=1,
)
metrics_dict = evaluate_fn(models_data, testing_dataloader, DEVICE)
metrics_df = pd.DataFrame(metrics_dict, index=[0])
logger.report_table(title=input_data, series="Metrics", table_plot=metrics_df)
task.close()
