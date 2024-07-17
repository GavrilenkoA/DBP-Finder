import argparse

import clearml
import pandas as pd
import torch
from clearml import Logger, Task
from data_prepare import form_test_kindom, prepare_test
from torch.utils.data import DataLoader
from torch_utils import (SequenceDataset, collect_logits_labels, evaluate_fn,
                         load_models)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("test_set", type=str, help="A string argument")
parser.add_argument("taxon", type=str, help="A string argument")
args = parser.parse_args()

clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name=f"Performance_{args.test_set}_{args.taxon}",
    output_uri=True,
)
logger = Logger.current_logger()


def evaluate_models_on_test_set(test_set, taxon):
    # Load models
    models = load_models()

    # Prepare test data
    test_df = form_test_kindom(test_set, taxon)
    test_df = prepare_test(
        test_df,
        embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{test_set}_2d.h5",
    )

    # Create dataloader
    testing_set = SequenceDataset(test_df)
    testing_dataloader = DataLoader(
        testing_set,
        num_workers=1,
        shuffle=False,
        batch_size=1,
    )

    # Evaluate models
    all_logits, all_labels = collect_logits_labels(models, testing_dataloader, DEVICE)
    metrics = evaluate_fn(all_labels, all_logits)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics, index=[0])

    # Log results
    logger.report_table(title=test_set, series=taxon, table_plot=metrics_df)
    task.close()
