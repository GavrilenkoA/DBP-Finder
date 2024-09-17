import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
import numpy as np
import pandas as pd
import copy


from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed

from peft import (
    inject_adapter_in_model,
    LoraConfig,
)

from evaluate import load
from datasets import Dataset

import random


from data_prepare import make_folds


accuracy_metric = load("accuracy")
f1_metric = load("f1")
matthews_metric = load("matthews_correlation")
precision_metric = load("precision")
recall_metric = load("recall")
roc_auc_metric = load("roc_auc")

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %%
checkpoint = "ElnaggarLab/ankh-base"


df = pd.read_csv("../data/splits/train_p3_pdb2272.csv")
train_dfs, valid_dfs = make_folds(df)
my_train = train_dfs[0]
my_valid = valid_dfs[0]


class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=1):
        self.dropout_rate = dropout
        self.num_labels = num_labels


class T5EncoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, class_config.num_labels)

    def forward(self, hidden_states):
        hidden_states = torch.mean(hidden_states, dim=1)  # avg embedding

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5EncoderForSimpleSequenceClassification(T5PreTrainedModel):
    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = T5EncoderClassificationHead(config, class_config)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_T5_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    # Load model and tokenizer
    if "ankh" in checkpoint:
        model = T5EncoderModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    elif "prot_t5" in checkpoint:
        # possible to load the half precision model (thanks to @pawel-rezo for pointing that out)
        if half_precision and deepspeed:
            tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
            )
            model = T5EncoderModel.from_pretrained(
                "Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16
            )  # .to(torch.device('cuda')
        else:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    elif "ProstT5" in checkpoint:
        if half_precision and deepspeed:
            tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(
                checkpoint, torch_dtype=torch.float16
            )  # .to(torch.device('cuda')
        else:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    class_config = ClassConfig(num_labels=num_labels)
    class_model = T5EncoderForSimpleSequenceClassification(model.config, class_config)

    # Set encoder and embedding weights to checkpoint weights
    class_model.shared = model.shared
    class_model.encoder = model.encoder

    # Delete the checkpoint model
    model = class_model
    del class_model

    if full:
        return model, tokenizer

    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_Classfier\nTrainable Parameter: " + str(params))

    # lora modification
    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["q", "k", "v", "o"]
    )

    model = inject_adapter_in_model(peft_config, model)

    # Unfreeze the prediction head
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True

    # Print trainable Parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_LoRA_Classfier\nTrainable Parameter:" + str(params) + "\n")

    return model, tokenizer


def load_esm_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if half_precision and deepspeed:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, torch_dtype=torch.float16
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels
        )

    if full:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )

    model = inject_adapter_in_model(peft_config, model)

    # Unfreeze the prediction head
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True

    return model, tokenizer


ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


def create_dataset(tokenizer, seqs, labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset


# Main training fuction
def train_per_protein(
    checkpoint,  # checkpoint
    train_df,  # training data
    valid_df,  # validation data
    num_labels=2,  # 1 for regression, >1 for classification
    # effective training batch size is batch * accum
    # we recommend an effective batch size of 8
    batch=4,  # for training
    accum=2,  # gradient accumulation
    val_batch=16,  # batch size for evaluation
    epochs=10,  # training epochs
    lr=3e-4,  # recommended learning rate
    seed=42,  # random seed
    deepspeed=False,  # if gpu is large enough disable deepspeed for training speedup
    mixed=True,  # enable mixed precision training
    full=False,  # enable training of the full model (instead of LoRA)
    gpu=2,
):  # gpu selection (1 for first gpu)
    print("Model used:", checkpoint, "\n")

    # Correct incompatible training settings
    if "ankh" in checkpoint and mixed:
        print("Ankh models do not support mixed precision training!")
        print("switched to FULL PRECISION TRAINING instead")
        mixed = False

    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

    # Set all random seeds
    set_seeds(seed)

    # load model
    if "esm" in checkpoint:
        model, tokenizer = load_esm_model(
            checkpoint, num_labels, mixed, full, deepspeed
        )
    else:
        model, tokenizer = load_T5_model(checkpoint, num_labels, mixed, full, deepspeed)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    # train_df["sequence"]=train_df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
    # valid_df["sequence"]=valid_df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)

    # Add spaces between each amino acid for ProtT5 and ProstT5 to correctly use them
    if "Rostlab" in checkpoint:
        train_df["sequence"] = train_df.apply(
            lambda row: " ".join(row["sequence"]), axis=1
        )
        valid_df["sequence"] = valid_df.apply(
            lambda row: " ".join(row["sequence"]), axis=1
        )

    # Add <AA2fold> for ProstT5 to inform the model of the input type (amino acid sequence here)
    if "ProstT5" in checkpoint:
        train_df["sequence"] = train_df.apply(
            lambda row: "<AA2fold> " + row["sequence"], axis=1
        )
        valid_df["sequence"] = valid_df.apply(
            lambda row: "<AA2fold> " + row["sequence"], axis=1
        )

    train_set = create_dataset(
        tokenizer, list(train_df["sequence"]), list(train_df["label"])
    )
    valid_set = create_dataset(
        tokenizer, list(valid_df["sequence"]), list(valid_df["label"])
    )

    # Huggingface Trainer arguments
    args = TrainingArguments(
        output_dir="ankh-lora-finetuned",
        logging_strategy="epoch",
        eval_strategy="epoch",  # Ensure evaluation occurs each epoch
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=mixed,
        weight_decay=0.01,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predicted_labels = np.argmax(predictions, axis=1)

        accuracy = accuracy_metric.compute(
            predictions=predicted_labels, references=labels
        )
        f1 = f1_metric.compute(predictions=predicted_labels, references=labels)
        matthews = matthews_metric.compute(
            predictions=predicted_labels, references=labels
        )
        precision = precision_metric.compute(
            predictions=predicted_labels, references=labels
        )
        recall = recall_metric.compute(predictions=predicted_labels, references=labels)

        metrics = {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"],
            "matthews_correlation": matthews["matthews_correlation"],
            "precision": precision["precision"],
            "recall": recall["recall"],
        }

        print("Metrics computed:", metrics)
        return metrics

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # Train model
    trainer.train()

    # Save model and tokenizer manually
    model_save_path = os.path.join("ankh-lora-finetuned", "final_model")
    tokenizer_save_path = os.path.join("ankh-lora-finetuned", "final_tokenizer")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"Model saved to {model_save_path}")
    print(f"Tokenizer saved to {tokenizer_save_path}")

    return tokenizer, model, trainer.state.log_history


tokenizer, model, history = train_per_protein(
    checkpoint,
    my_train,
    my_valid,
    num_labels=2,
    batch=10,
    val_batch=10,
    accum=2,
    epochs=7,
    seed=42,
    mixed=False,
    deepspeed=False,
)
