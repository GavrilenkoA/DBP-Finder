import argparse
import logging
import os

import ankh
import clearml
import numpy as np
import torch
import yaml
from clearml import Logger, Task
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_prepare import get_embed_clustered_df, make_folds
from utils import (CustomBatchSampler, SequenceDataset, custom_collate_fn,
                   train_fn, validate_fn)


def main():
    parser = argparse.ArgumentParser(description="Train DBP-Finder model")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="../../../../ssd2/dbp_finder/ankh_embeddings/train_2d.h5",
        help="Path to the HDF5 file containing embeddings"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/splits/train_p3.csv",
        help="Path to the CSV file containing training data"
    )
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="checkpoints/DBP-Finder",
        help="Path to save the best model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DBP-Finder-config.yml",
        help="Path to the configuration YAML file"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract model and training configuration
    input_dim = config["model_config"]["input_dim"]
    nhead = config["model_config"]["nhead"]
    hidden_dim = config["model_config"]["hidden_dim"]
    num_hidden_layers = config["model_config"]["num_hidden_layers"]
    num_layers = config["model_config"]["num_layers"]
    kernel_size = config["model_config"]["kernel_size"]
    dropout = config["model_config"]["dropout"]
    pooling = config["model_config"]["pooling"]

    epochs = config["training_config"]["epochs"]
    lr = float(config["training_config"]["lr"])
    factor = config["training_config"]["factor"]
    patience = config["training_config"]["patience"]
    min_lr = float(config["training_config"]["min_lr"])
    batch_size = config["training_config"]["batch_size"]
    seed = config["training_config"]["seed"]
    num_workers = config["training_config"]["num_workers"]
    weight_decay = float(config["training_config"]["weight_decay"])

    # Set environment variables for GPU usage
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    DEVICE = "cuda"

    def set_seed(seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Set the random seed
    set_seed(seed)

    # Load the data
    df = get_embed_clustered_df(
        embedding_path=args.embedding_path,
        csv_path=args.csv_path,
    )

    # Create training and validation folds
    train_folds, valid_folds = make_folds(df)

    clearml.browser_login()
    task = Task.init(
        project_name="DBPs_search",
        task_name="Training Ankh head on the full data",
        output_uri=True,
    )
    logger = Logger.current_logger()
    task.connect_configuration(config)

    thresholds = {}
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
        valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )

        # Initialize the model
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
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )

        best_val_loss = float("inf")
        model_path = args.best_model_path + f"_{i}.pth"
        for epoch in range(epochs):
            train_loss = train_fn(model, train_dataloader, optimizer, DEVICE)
            valid_loss, metrics_dict, threshold = validate_fn(
                model, valid_dataloader, scheduler, DEVICE
            )

            # Log the metrics
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

            # Save the best model
            if valid_loss < best_val_loss:
                thresholds[i] = threshold
                torch.save(model.state_dict(), model_path)

                training_log = (
                    f"model {i} on epoch {epoch} with validation loss: {valid_loss}"
                )
                threshold_log = f"best_threshold: {threshold} on epoch {epoch} of model {i}"

                logger.report_text(training_log, level=logging.DEBUG, print_console=False)
                logger.report_text(threshold_log, level=logging.DEBUG, print_console=False)
                best_val_loss = valid_loss

    task.upload_artifact(name="thresholds k folds", artifact_object=thresholds)
    task.close()


if __name__ == "__main__":
    main()
