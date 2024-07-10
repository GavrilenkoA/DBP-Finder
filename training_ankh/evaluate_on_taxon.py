import argparse
import yaml
import pandas as pd
import ankh
import torch
import clearml
from clearml import Logger, Task
from torch.utils.data import DataLoader
from torch_utils import SequenceDataset, evaluate_fn
from data_prepare import form_test_kindom, prepare_test


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("test_set", type=str, help="A string argument")
parser.add_argument("taxon", type=str, help="A string argument")
args = parser.parse_args()

test_df = form_test_kindom(args.test_set, args.taxon)
clearml.browser_login()
task = Task.init(
    project_name="DBPs_search",
    task_name=f"Performance_{args.test_set}_{args.taxon}",
    output_uri=True,
)
logger = Logger.current_logger()
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


input_dim = config["model_config"]["input_dim"]
nhead = config["model_config"]["nhead"]
hidden_dim = config["model_config"]["hidden_dim"]
num_hidden_layers = config["model_config"]["num_hidden_layers"]
num_layers = config["model_config"]["num_layers"]
kernel_size = config["model_config"]["kernel_size"]
dropout = config["model_config"]["dropout"]
pooling = config["model_config"]["pooling"]
num_workers = config["training_config"]["num_workers"]

models = []
for i in range(5):  # Assuming we have 5 models
    binary_classification_model = ankh.ConvBertForBinaryClassification(
        input_dim=input_dim,
        nhead=nhead,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        pooling=pooling,
    )

    path_model = f"checkpoints/{args.test_set}_best_model_{i}.pth"
    binary_classification_model.load_state_dict(torch.load(path_model))
    binary_classification_model.eval()  # Set the model to evaluation mode
    models.append(binary_classification_model)


test_df = prepare_test(
    test_df,
    embedding_path=f"../../../../ssd2/dbp_finder/ankh_embeddings/{args.test_set}_2d.h5",
)
testing_set = SequenceDataset(test_df)
testing_dataloader = DataLoader(
    testing_set,
    num_workers=num_workers,
    shuffle=False,
    batch_size=1,
)
metrics = evaluate_fn(models, testing_dataloader, DEVICE)
metrics_df = pd.DataFrame(metrics, index=[0])
logger.report_table(title=args.test_set, series=args.taxon, table_plot=metrics_df)
task.close()
