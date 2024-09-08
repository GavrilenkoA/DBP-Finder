from transformers import EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
from clearml import Task
from clearml import OutputModel
from data_prepare import make_folds
from HF_utils import ClearMLCallback


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_checkpoint = "ElnaggarLab/ankh-base"
model_name = model_checkpoint.split("/")[-1]
task = Task.init(project_name="DBPs_search", task_name=f"lora q, k, v {model_name} pdb2272")


def model_init(model_checkpoint):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=16,
        target_modules=["q", "k", "v"],
        lora_dropout=0.1,
        bias="all",
    )

    model = get_peft_model(base_model, config)
    return model


model = model_init(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size = 64

# Prepare datasets
df = pd.read_csv("../data/splits/train_p3_pdb2272.csv")
train_dfs, valid_dfs = make_folds(df)
train = train_dfs[0]
valid = valid_dfs[0]

train_sequences = train["sequence"].tolist()
train_labels = train["label"].tolist()
valid_sequences = valid["sequence"].tolist()
valid_labels = valid["label"].tolist()

train_tokenized = tokenizer(train_sequences)
valid_tokenized = tokenizer(valid_sequences)

train_dataset = Dataset.from_dict(train_tokenized).add_column("labels", train_labels)
valid_dataset = Dataset.from_dict(valid_tokenized).add_column("labels", valid_labels)

# Metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")
matthews_metric = load("matthews_correlation")
precision_metric = load("precision")
recall_metric = load("recall")
roc_auc_metric = load("roc_auc")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Compute each metric
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    matthews = matthews_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    roc_auc = roc_auc_metric.compute(predictions=predictions, references=labels)

    metrics = {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "matthews_correlation": matthews["matthews_correlation"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "roc_auc": roc_auc["roc_auc"],
    }

    print("Metrics computed:", metrics)  # Debugging: Print metrics
    return metrics


args = TrainingArguments(
    output_dir=f"{model_name}-lora-finetuned",
    evaluation_strategy="epoch",  # Ensure evaluation occurs each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Change this if you prefer another metric like accuracy
    greater_is_better=False,
    logging_dir='./logs',  # Ensure logging directory is set
    logging_strategy="steps",
    logging_steps=10,  # Log metrics every 10 steps
    report_to="clearml",  # Avoids issues with not setting a reporting tool
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[],
)
trainer.train()

# Close the ClearML task
task.close()
