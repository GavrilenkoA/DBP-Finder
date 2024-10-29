import io

import clearml
import pandas as pd
import torch
from clearml import Logger, Task

from dataset import InferenceDataset as IF_Dataset
from dataset import dataloader_prepare
from train_ankh_utils import ensemble_inference
from utils import load_lora_models, load_thresholds

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = "pdb20000"

clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="not-annotated-GO:0003676-DBP-Finder-v1",
    output_uri=False,
)
logger = Logger.current_logger()


def predict(test_data):
    models, tokenizer = load_lora_models(prefix_name="ankh-base-lora-finetuned/DBP-Finder_")
    thresholds = load_thresholds(model_name="lora_ankh_train_p3")
    test_df = pd.read_csv("../data/not_annotated/not_annotated_GO:0003676.csv")
    dataloader = dataloader_prepare(
        test_df, tokenizer, dataset_class=IF_Dataset, shuffle=False, labels_flag=False
    )
    predictions_df = ensemble_inference(models, dataloader, thresholds, DEVICE)

    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    task.upload_artifact(name=test_data, artifact_object=csv_buffer)
    task.close()


predict(test_data)
