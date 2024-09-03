import random
from torch.utils.data import Dataset
from transformers import TrainerCallback
from clearml import Task


class ClearMLCallback(TrainerCallback):
    def __init__(self, task_name="Training Task"):
        self.task = Task.init(project_name="DBPs_search", task_name=task_name, task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.task.upload_scalar(key, value, step=state.global_step)

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.task.upload_scalar(f"eval_{key}", value, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.task.close()


class SortedDataset(Dataset):
    def __init__(self, tokenized_data, labels, shuffle=True, batch_size=64):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = labels
        self.lengths = [len(ids) for ids in self.input_ids]

        # Sort by length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda k: self.lengths[k])
        self.input_ids = [self.input_ids[i] for i in sorted_indices]
        self.attention_mask = [self.attention_mask[i] for i in sorted_indices]
        self.labels = [self.labels[i] for i in sorted_indices]

        # Chunk the dataset into batches of similar length
        self.batch_size = batch_size
        self.batched_indices = [sorted_indices[i:i + batch_size] for i in range(0, len(sorted_indices), batch_size)]

        if shuffle:
            # Shuffle the batches
            random.shuffle(self.batched_indices)

            # Shuffle within each batch
            for batch in self.batched_indices:
                random.shuffle(batch)

        # Flatten the batched indices back to a single list
        self.shuffled_indices = [i for batch in self.batched_indices for i in batch]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Access the shuffled index
        real_idx = self.shuffled_indices[idx]
        return {
            'input_ids': self.input_ids[real_idx],
            'attention_mask': self.attention_mask[real_idx],
            'labels': self.labels[real_idx],
        }
