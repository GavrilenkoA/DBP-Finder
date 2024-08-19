import logging

import pandas as pd
import ankh
import clearml
import optuna
import torch
import torch.nn as nn
from clearml import Logger, Task
from data_prepare import get_embed_clustered_df, make_folds
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_utils import (
    CustomBatchSampler,
    SequenceDataset,
    custom_collate_fn,
    train_fn,
    validate_fn,
)

clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="Optuna search pdb2272",
    output_uri=False,
)
logger = Logger.current_logger()

df = get_embed_clustered_df(
    embedding_path="../data/embeddings/ankh_embeddings/train_p3_2d.h5",
    csv_path="../data/splits/train_pdb2272.csv",
)
train_folds, valid_folds = make_folds(df)


def objective(trial):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    pooling = trial.suggest_categorical("pooling", ["max", "avg"])
    hidden_dim = trial.suggest_int("hidden_dim", 1425, 2120)
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 2)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    nhead = trial.suggest_int("nhead", 3, 6)

    model = ankh.ConvBertForBinaryClassification(
        input_dim=1536,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=num_layers,
        kernel_size=7,
        dropout=dropout,
        pooling=pooling,
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    fold_losses = []
    for fold in range(len(train_folds)):
        train_df = train_folds[fold]
        valid_df = valid_folds[fold]

        train_dataset = SequenceDataset(train_df)
        train_sampler = CustomBatchSampler(train_dataset, batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=4,
            batch_sampler=train_sampler,
            collate_fn=custom_collate_fn,
        )

        valid_dataset = SequenceDataset(valid_df)
        valid_sampler = CustomBatchSampler(valid_dataset, batch_size)
        valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=4,
            batch_sampler=valid_sampler,
            collate_fn=custom_collate_fn,
        )

        best_val_loss = float("inf")
        for epoch in range(11):
            train_fn(model, train_dataloader, optimizer, DEVICE)
            valid_loss, metrics = validate_fn(
                model, valid_dataloader, scheduler, DEVICE
            )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            metrics_df = pd.DataFrame(metrics, index=["-"])
            logger.report_table(
                title=f"Fold {fold}", series=f"Epoch {epoch}", table_plot=metrics_df
            )

        fold_losses.append(best_val_loss)
    avg_loss = sum(fold_losses) / len(fold_losses)
    return avg_loss


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    trial = study.best_trial
    for key, value in trial.params.items():
        message = f"{key}: {value}"
        logger.report_text(message, level=logging.DEBUG, print_console=True)

    logger.report_text(
        f"Best trial number: {trial.number}", level=logging.INFO, print_console=True
    )
    logger.report_text(
        f"Best trial value (loss): {trial.value}",
        level=logging.INFO,
        print_console=True,
    )
    task.close()


if __name__ == "__main__":
    main()
