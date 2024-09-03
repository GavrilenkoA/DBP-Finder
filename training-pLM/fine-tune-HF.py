from transformers import EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
import os
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from data_prepare import make_folds
from HF_utils import ClearMLCallback


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

accelerator = Accelerator()


def model_init(model_checkpoint):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Changed to SEQ_CLS for sequence classification
        r=16,
        lora_alpha=16,
        target_modules=["q", "k", "v"],
        lora_dropout=0.1,
        bias="all",
    )

    lora_model = get_peft_model(base_model, config)
    return accelerator.prepare(lora_model)


model_checkpoint = "ElnaggarLab/ankh-base"
model = model_init(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size = 64


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
train_dataset = Dataset.from_dict(train_tokenized)
valid_dataset = Dataset.from_dict(valid_tokenized)

train_dataset = train_dataset.add_column("labels", train_labels)
valid_dataset = valid_dataset.add_column("labels", valid_labels)

train_dataset = accelerator.prepare(train_dataset)
valid_dataset = accelerator.prepare(valid_dataset)

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

    return metrics


model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    output_dir=f"{model_name}-Lora-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

clearml_callback = ClearMLCallback(task_name=f"Lora fine-tuning {model_name}, pdb2272 train data")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[clearml_callback],
)
trainer.train()
