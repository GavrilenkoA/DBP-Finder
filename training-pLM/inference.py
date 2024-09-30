import clearml
import pandas as pd
import json
import io
import torch
from clearml import Logger, Task
from data_prepare import form_test_kingdom, get_embed_clustered_df, prepare_test
from torch.utils.data import DataLoader
from dataset import dataloader_prepare, PredictionDataset, SequenceDatasetWithID
from utils import InferenceDataset, SequenceDataset, inference, load_ff_ankh, load_lora_models, load_models, evaluate_ensemble_based_on_threshold, inference_ensemble_based_on_threshold
from train_ankh_utils import ensemble_predict, ensemble_inference

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = "test_trembl_proteinfer"
clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name=f"lora-ankh-Sequence_with_muts",
    output_uri=False,
)
logger = Logger.current_logger()


def load_thresolds(model_name="lora_ankh_train_p3"):
    with open("thresholds.json", "r") as json_file:
        thresholds = json.load(json_file)

    thresholds = thresholds[model_name]
    upd_thresholds = {}
    for key in thresholds:
        int_key = int(key)
        upd_thresholds[int_key] = thresholds[key]
    return upd_thresholds


def ankh_head_inference(test_data):
    models = load_models(prefix_name="checkpoints/models/DBP-Finder_batch_64", config_path="config.yml")
    thresholds = load_thresolds(model_name="64_epoch_train_train_p3")
    test_df = get_embed_clustered_df(
        embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{test_data}_2d.h5",
        csv_path=f"../data/splits/{test_data}.csv")

    testing_dataset = SequenceDataset(test_df)
    testing_dataloader = DataLoader(
        testing_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1)

    inference_dataset = InferenceDataset(test_df)  # сделать как один класс
    inference_dataloader = DataLoader(
        inference_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )

    metrics_dict = evaluate_ensemble_based_on_threshold(models, testing_dataloader, thresholds, DEVICE)
    metrics_df = pd.DataFrame(metrics_dict, index=[test_data])
    logger.report_table(title="Metrics", series=test_data, table_plot=metrics_df)
    predictions_df = inference_ensemble_based_on_threshold(models, inference_dataloader, thresholds, DEVICE)
    predictions_df = inference(models, inference_dataloader, DEVICE)
    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    task.upload_artifact(name=f"DBP-Finder_{test_data}", artifact_object=csv_buffer)
    task.close()


def ankh_inference(test_data):
    models, tokenizer = load_lora_models(prefix_name="ankh-base-lora-finetuned/DBP-Finder_", num_models=5)
    # models, tokenizer = load_ff_ankh()
    thresholds = load_thresolds(model_name="lora_ankh_train_p3")
    # thresholds = load_thresolds(model_name="ff_ankh_train_p3")
    # test_df = pd.read_csv(f"../data/splits/{test_data}.csv")
    test_df = pd.read_csv("../data/test_cases/Sequence_with_muts.csv", sep=";")
    dataloader = dataloader_prepare(test_df, tokenizer, dataset_class=PredictionDataset, shuffle=False, labels_flag=False)
    # metrics_df, predictions_df = ensemble_predict(models, dataloader, thresholds, DEVICE)
    predictions_df = ensemble_inference(models, dataloader, thresholds, DEVICE)

    # logger.report_table(title="Metrics", series=test_data, table_plot=metrics_df)
    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    task.upload_artifact(name="lora-ankh-Sequence_with_muts", artifact_object=csv_buffer)
    task.close()


ankh_inference(test_data)
