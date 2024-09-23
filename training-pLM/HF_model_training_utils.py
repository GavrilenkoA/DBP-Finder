from tqdm import tqdm
import torch
import numpy as np
from utils import calculate_metrics, find_best_threshold


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


def validate_fn(model, valid_dataloader, scheduler, DEVICE):
    model.eval()
    loss = 0.0
    all_labels = []
    scores = []

    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            labels = labels.unsqueeze(1)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
