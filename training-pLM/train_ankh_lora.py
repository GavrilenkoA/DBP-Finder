import argparse
import logging
import os

import clearml
import numpy as np
import pandas as pd
import torch
import yaml
from clearml import Logger, Task
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_prepare import make_folds
from dataset import CustomBatchSampler, SequenceDataset, collate_fn
from train_ankh_utils import train_fn, validate_fn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda"


def model_init(model_checkpoint, config_lora):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=1
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config_lora["r"],
        lora_alpha=config_lora["lora_alpha"],
        target_modules=config_lora["target_modules"],
        lora_dropout=config_lora["lora_dropout"],
        bias=config_lora["bias"],
    )

    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def df_prepare(csv_path):
    df = pd.read_csv(csv_path)
    train_folds, valid_folds = make_folds(df)
    return train_folds, valid_folds


def dataset_prepare(
    data: pd.DataFrame, tokenizer, batch_size, num_workers, shuffle=False
):
    dataset = SequenceDataset(data)
    batch_sampler = CustomBatchSampler(dataset, batch_size, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, tokenizer, labels_flag=True),
    )
    return dataloader


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Train Ankh model with LoRA.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/splits/train_p3.csv",
        help="Path to the CSV file containing training data"
    )
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="ankh-base-lora-finetuned/DBP-Finder",
        help="Path or name of the pre-trained model checkpoint"
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="lora_config.yml",
        help="Path to the LoRA configuration YAML file"
    )
    args = parser.parse_args()

    # Load LoRA configuration
    with open(args.lora_config, "r") as f:
        config = yaml.safe_load(f)

    clearml.browser_login()
    task = Task.init(
        project_name="DBPs_search",
        task_name="Ankh LoRA train",
        output_uri=True,
    )

    logger = Logger.current_logger()
    task.connect_configuration(config)

    epochs = config["training_config"]["epochs"]
    lr = float(config["training_config"]["lr"])
    factor = config["training_config"]["factor"]
    patience = config["training_config"]["patience"]
    min_lr = float(config["training_config"]["min_lr"])
    batch_size = config["training_config"]["batch_size"]
    seed = config["training_config"]["seed"]
    num_workers = config["training_config"]["num_workers"]
    weight_decay = float(config["training_config"]["weight_decay"])
    model_checkpoint = config["training_config"]["model_checkpoint"]
    config_lora = config["lora_config"]

    set_seed(seed)

    best_thresholds = {}
    train_folds, valid_folds = df_prepare(args.csv_path)
    for i in range(len(train_folds)):
        train = train_folds[i]
        valid = valid_folds[i]

        model, tokenizer = model_init(model_checkpoint, config_lora)

        train_dataloader = dataset_prepare(
            train, tokenizer, batch_size, num_workers, shuffle=True
        )

        valid_dataloader = dataset_prepare(
            valid, tokenizer, batch_size, num_workers, shuffle=False
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
            valid_loss, metrics_dict, best_threshold = validate_fn(
                model, valid_dataloader, scheduler, DEVICE
            )
            torch.cuda.empty_cache()

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
                best_thresholds[i] = best_threshold
                torch.save(model.state_dict(), model_path)
                training_log = (
                    f"model {i} on epoch {epoch} with validation loss: {valid_loss}"
                )
                threshold_log = (
                    f"best_threshold: {best_threshold} on epoch {epoch} of model {i}"
                )

                logger.report_text(
                    training_log, level=logging.DEBUG, print_console=False
                )
                logger.report_text(
                    threshold_log, level=logging.DEBUG, print_console=False
                )
                best_val_loss = valid_loss

    task.upload_artifact(name="best_thresholds", artifact_object=best_thresholds)
    task.close()


if __name__ == "__main__":
    main()
