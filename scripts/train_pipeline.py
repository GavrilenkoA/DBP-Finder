import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import clearml
from clearml import Task as clearml_task
from clearml import Logger
import joblib
import os
import argparse

from utils import SEED
from train_src import (
    make_lama_df,
    merge_embed,
    Metrics,
    form_Xy,
    plot_roc_curve,
    filter_test_by_kingdom,
)


parser = argparse.ArgumentParser(description="DBP-finder training")
parser.add_argument(
    "input_test", type=str, help="A required integer positional argument"
)
parser.add_argument(
    "--kingdom", type=str, default=None, help="Specify kingdom for testing"
)
args = parser.parse_args()


clearml.browser_login()
task_ml = clearml_task.init(
    project_name="DBPs_search",
    task_name=f"{args.input_test}_{args.kingdom}",
    output_uri=True,
)
logger = Logger.current_logger()


def collect_train():
    train = pd.read_csv(f"data/ready_data/train_{args.input_test}.csv")
    train = merge_embed(train, "train_p2")
    X_train, y_train, clusters_train = form_Xy(train, clusters="Yes")
    df_train = make_lama_df(X_train, y_train, clusters_train)
    return df_train, y_train


def collect_test():
    basename = args.input_test.split("_")[0]
    test = pd.read_csv(f"data/embeddings/input_csv/{basename}.csv")
    if args.kingdom is not None:
        test = filter_test_by_kingdom(test, args.input_test, args.kingdom)
    test = merge_embed(test, basename)
    X_test, y_test = form_Xy(test)
    df_test = make_lama_df(X_test, y_test)
    return df_test, y_test


def train_func():
    df_train, y_train = collect_train()
    roles = {"target": "label", "group": "cluster"}
    task = Task("binary")
    automl = TabularAutoML(task=task, reader_params={"random_state": SEED})

    oof_pred = automl.fit_predict(df_train, roles=roles)

    valid_prob = oof_pred.data[:, 0]
    valid_pred = (valid_prob > 0.5) * 1
    metrics = Metrics(y_train, valid_pred, valid_prob, "valid")
    valid_metrics = metrics.get_metrics()
    logger.report_table(
        title="Validation metrics", series="pandas DataFrame", table_plot=valid_metrics
    )

    joblib.dump(automl, f"models/{args.input_test}.pkl")


def eval_func():
    df_test, y_test = collect_test()
    automl = joblib.load(f"models/{args.input_test}.pkl")
    test_pred = automl.predict(df_test)
    test_prob = test_pred.data.reshape(
        -1,
    )

    test_pred = (test_pred.data[:, 0] > 0.5) * 1
    metrics = Metrics(y_test, test_pred, test_prob, args.input_test)
    test_metrics = metrics.get_metrics()
    logger.report_table(
        title="Test metrics", series="pandas DataFrame", table_plot=test_metrics
    )
    plot_roc_curve(y_test, test_prob, args.input_test)


def main():
    if not os.path.exists(f"models/{args.input_test}.pkl"):
        train_func()

    eval_func()


if __name__ == "__main__":
    main()
