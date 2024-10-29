from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from tqdm import tqdm

from utils import calculate_metrics, find_best_threshold, plot_roc_auc


def train_fn(model, train_dataloader, optimizer, DEVICE):
    model.train()
    loss = 0.0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        output.loss.backward()
        optimizer.step()

        loss += output.loss.item()

    epoch_loss = loss / len(train_dataloader)
    return epoch_loss


def validate_fn(model, valid_dataloader, scheduler, DEVICE):
    model.eval()
    loss = 0.0
    all_labels = []
    scores = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            labels = labels.unsqueeze(1)

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss += output.loss.item()

            score = torch.sigmoid(output.logits)

            scores.extend(score.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    best_threshold = find_best_threshold(all_labels, scores)
    all_preds = (np.array(scores) >= best_threshold).astype(int).tolist()

    epoch_loss = loss / len(valid_dataloader)
    scheduler.step(epoch_loss)
    metrics_dict = calculate_metrics(scores, all_labels, all_preds)
    return epoch_loss, metrics_dict, best_threshold


def ensemble_predict(models, dataloader, thresholds, DEVICE):
    all_labels = []
    all_identifiers = []
    score_per_model = defaultdict(list)
    prediction_per_model = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Get identifiers if they exist
            identifiers = batch.get("identifiers", None)
            if identifiers is not None:
                all_identifiers.extend(identifiers)

            all_labels.extend(labels.cpu().numpy().flatten())

            for i, model in models.items():
                model.eval().to(DEVICE)
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                score = torch.sigmoid(output.logits).cpu().numpy().flatten()
                score_per_model[i].extend(score)

                pred = (score >= thresholds[i]).astype(int)
                prediction_per_model[i].extend(pred)

    # Convert defaultdicts to lists
    prediction_per_model = list(prediction_per_model.values())
    score_per_model = list(score_per_model.values())

    # Majority voting for predictions
    predictions = mode(prediction_per_model, axis=0)[0].tolist()
    scores = np.mean(score_per_model, axis=0).tolist()

    # Plot ROC-AUC and calculate metrics
    plot_roc_auc(all_labels, scores)
    metrics_dict = calculate_metrics(scores, all_labels, predictions)
    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    # Create a DataFrame with predictions, scores, labels, and identifiers
    predictions_df = pd.DataFrame(
        {
            "identifier": all_identifiers
            if all_identifiers
            else range(len(all_labels)),  # Add fallback if no identifiers
            "true_label": all_labels,
            "predicted_score": scores,
            "predicted_label": predictions,
        }
    )

    return metrics_df, predictions_df


def ensemble_inference(models, dataloader, thresholds, DEVICE):
    all_identifiers = []
    score_per_model = defaultdict(list)
    prediction_per_model = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Get identifiers if they exist
            identifiers = batch.get("identifiers", None)
            if identifiers is not None:
                all_identifiers.extend(identifiers)

            for i, model in models.items():
                model.eval().to(DEVICE)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                score = torch.sigmoid(output.logits).cpu().numpy().flatten()
                score_per_model[i].extend(score)

                pred = (score >= thresholds[i]).astype(int)
                prediction_per_model[i].extend(pred)

    # Convert defaultdicts to lists
    prediction_per_model = list(prediction_per_model.values())
    score_per_model = list(score_per_model.values())

    # Majority voting for predictions
    predictions = mode(prediction_per_model, axis=0)[0].tolist()
    scores = np.mean(score_per_model, axis=0).tolist()

    # Create a DataFrame with predictions, scores, labels, and identifiers
    predictions_df = pd.DataFrame(
        {
            "identifier": all_identifiers
            if all_identifiers
            else range(len(scores)),  # Add fallback if no identifiers
            "predicted_score": scores,
            "predicted_label": predictions,
        }
    )
    return predictions_df


def get_learning_rate(optimizer):
    """Retrieve the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
