import ankh
import torch
import optuna
from torch.optim import AdamW
from torch.utils.data import DataLoader
import clearml
import logging
from clearml import Logger, Task
from torch_utils import (
    SequenceDataset,
    CustomBatchSampler,
    custom_collate_fn,
    train_fn,
    validate_fn,
)
from data_prepare import prepare_embed_df, make_folds


df = prepare_embed_df()
train, valid = make_folds(df)


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

    binary_classification_model = ankh.ConvBertForBinaryClassification(
        input_dim=1536,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=num_layers,
        kernel_size=7,
        dropout=dropout,
        pooling=pooling,
    )

    binary_classification_model = binary_classification_model.to(DEVICE)
    optimizer = AdamW(binary_classification_model.parameters(), lr=lr)

    train_dataset = SequenceDataset(train)
    train_sampler = CustomBatchSampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_sampler=train_sampler,
        collate_fn=custom_collate_fn,
    )

    valid_dataset = SequenceDataset(valid)
    valid_sampler = CustomBatchSampler(valid_dataset, batch_size)
    valid_dataloader = DataLoader(
        valid_dataset,
        num_workers=4,
        batch_sampler=valid_sampler,
        collate_fn=custom_collate_fn,
    )

    best_val_loss = float("inf")
    for _ in range(8):
        train_fn(binary_classification_model, train_dataloader, optimizer, DEVICE)
        valid_loss = validate_fn(
            binary_classification_model, valid_dataloader, DEVICE
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss

    return best_val_loss


def main():
    clearml.browser_login()
    task = Task.init(
        project_name="DBPs_search",
        task_name="Optuna search fold 0 pdb2272",
        output_uri=False)

    logger = Logger.current_logger()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25)
    trial = study.best_trial
    for key, value in trial.params.items():
        message = f"{key}: {value}"
        logger.report_text(message, level=logging.DEBUG, print_console=True)

    logger.report_text(f"Best trial number: {trial.number}", level=logging.INFO, print_console=True)
    logger.report_text(f"Best trial value (loss): {trial.value}", level=logging.INFO, print_console=True)

    task.close()


if __name__ == "__main__":
    main()
