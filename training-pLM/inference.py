import clearml
import pandas as pd
import torch
from clearml import Logger, Task
from data_prepare import form_test_kingdom, get_embed_clustered_df, prepare_test
from torch.utils.data import DataLoader
from torch_utils import InferenceDataset, SequenceDataset, evaluate_fn, inference, load_models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = "test_trembl_proteinfer"

clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name=test_data,
    output_uri=False,
)
logger = Logger.current_logger()


def evaluate_on_taxon(test_data) -> None:
    models = load_models(num_models=f"checkpoints/{test_data}_best_model_")

    kingdoms = ['Archaea', 'Viruses', 'Bacteria', 'Metazoa', 'Fungi', 'Protists',
                'Viridiplantae']

    for kingdom in kingdoms:
        test_df = form_test_kingdom(test_data, kingdom)
        test_df = prepare_test(test_df,
                               embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{test_data}.h5")

        testing_set = SequenceDataset(test_df)
        testing_dataloader = DataLoader(
            testing_set,
            num_workers=1,
            shuffle=False,
            batch_size=1,
        )
        metrics_dict = evaluate_fn(models, testing_dataloader, DEVICE)
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        logger.report_table(title=test_data, series=kingdom, table_plot=metrics_df)

    task.close()


def evaluate_on_test(test_data: str) -> None:
    models = load_models()
    test_df = get_embed_clustered_df(
        embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{test_data}_2d.h5",
        csv_path=f"../data/splits/{test_data}.csv")

    testing_dataset = SequenceDataset(test_df)
    testing_dataloader = DataLoader(
        testing_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1)

    inference_dataset = InferenceDataset(test_df)
    inference_dataloader = DataLoader(
        inference_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )

    metrics_dict = evaluate_fn(models, testing_dataloader, DEVICE)
    metrics_df = pd.DataFrame(metrics_dict, index=[test_data])
    logger.report_table(title="Metrics", series=test_data, table_plot=metrics_df)
    predictions_df = inference(models, inference_dataloader, DEVICE)
    predictions_df.to_csv("../data/prediction/proteinfer_predictions.csv", index=False)
    task.close()


evaluate_on_test(test_data)
