import clearml
import pandas as pd
import torch
from clearml import Logger, Task
from data_prepare import form_test_kingdom, get_embed_clustered_df, prepare_test
from torch.utils.data import DataLoader
from torch_utils import SequenceDataset, evaluate_fn, load_models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_data = input()

clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name="test_trembl_MF",
    output_uri=False,
)
logger = Logger.current_logger()


def evaluate_on_taxon(input_data) -> None:
    models = load_models(num_models=f"checkpoints/{input_data}_best_model_")

    kingdoms = ['Archaea', 'Viruses', 'Bacteria', 'Metazoa', 'Fungi', 'Protists',
                'Viridiplantae']

    for kingdom in kingdoms:
        test_df = form_test_kingdom(input_data, kingdom)
        test_df = prepare_test(test_df,
                               embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{input_data}.h5")

        testing_set = SequenceDataset(test_df)
        testing_dataloader = DataLoader(
            testing_set,
            num_workers=1,
            shuffle=False,
            batch_size=1,
        )
        metrics_dict = evaluate_fn(models, testing_dataloader, DEVICE)
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        logger.report_table(title=input_data, series=kingdom, table_plot=metrics_df)

    task.close()


def evaluate_on_test() -> None:
    models = load_models()
    test_df = get_embed_clustered_df(
        embedding_path="../../../../ssd2/dbp_finder/ankh_embeddings/test_trembl_MF_2d.h5",
        csv_path="../data/splits/test_trembl_MF.csv")

    testing_set = SequenceDataset(test_df)
    testing_dataloader = DataLoader(
        testing_set,
        num_workers=1,
        shuffle=False,
        batch_size=1)

    metrics_dict = evaluate_fn(models, testing_dataloader, DEVICE)
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    logger.report_table(title="trembl_proteinfer", series="Metrics", table_plot=metrics_df)
    task.close()


evaluate_on_test()
