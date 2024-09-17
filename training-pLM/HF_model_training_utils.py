from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_curve
from utils import calculate_metrics, plot_roc_auc


def train_fn(model, train_dataloader, optimizer, DEVICE):
    model.train()
    loss = 0.0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        output.loss.backward()
        optimizer.step()

        loss += output.loss.item()

    epoch_loss = loss / len(train_dataloader)
    return epoch_loss


def find_best_threshold(y_true: list[float], y_scores: list[float]) -> float:
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    return best_threshold


def validate_fn(model, valid_dataloader, scheduler, DEVICE):
    model.eval()
    loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            labels = labels.unsqueeze(1)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss += output.loss.item()

            prob = torch.sigmoid(output.logits)

            all_probs.extend(prob.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    best_threshold = find_best_threshold(all_labels, all_probs)
    all_preds = (np.array(all_probs) > best_threshold).astype(int).tolist()

    epoch_loss = loss / len(valid_dataloader)
    scheduler.step(epoch_loss)
    metrics_dict = calculate_metrics(all_probs, all_labels, all_preds)
    return epoch_loss, metrics_dict, best_threshold


def evaluate_fn(models, test_dataloader, DEVICE):
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            labels = labels.unsqueeze(1)

            assert len(labels) == 1, "Batch size should be 1"

            ens_logits = []

            for i in models:
                model = models[i].eval()
                model = model.to(DEVICE)
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ens_logits.append(output.logits)

            ens_logits = torch.stack(ens_logits, dim=0)
            ens_logits = torch.mean(ens_logits, dim=0)

            all_logits.append(ens_logits.cpu().numpy().item())
            all_labels.append(labels.cpu().numpy().item())

    all_logits_tensor = torch.tensor(np.array(all_logits))
    prob_score = torch.sigmoid(all_logits_tensor)
    all_preds = (prob_score > 0.5).float().tolist()
    metrics_dict = calculate_metrics(all_logits, all_labels, all_preds)
    plot_roc_auc(all_labels, all_logits)
    return metrics_dict
