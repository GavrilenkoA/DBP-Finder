import clearml
import pandas as pd
from clearml import Logger
from clearml import Task as clearml_task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from scripts.train_src import (Metrics, form_Xy, make_lama_df, merge_embed,
                               plot_roc_curve, reduce_train)
from scripts.utils import SEED

clearml.browser_login()

test_data = input()
model_name = "ankh"
identity = 0.5

task_ml = clearml_task.init(
    project_name="DBPs_search",
    task_name=f"{test_data}_identity_{identity}_model{model_name}",
    output_uri=True
)
logger = Logger.current_logger()

train = pd.read_csv("data/embeddings/input_csv/train_p2.csv")
test = pd.read_csv(f"data/embeddings/input_csv/{test_data}.csv")
clusters_data = pd.read_csv(f"data/ready_data/{test_data}_train_{identity}.csv")

train = reduce_train(clusters_data, train, test)
train = merge_embed(train, "train_p2")
X_train, y_train, clusters_train = form_Xy(train, clusters="Yes")
df_train = make_lama_df(X_train, y_train, clusters_train)

test = merge_embed(test, test_data)
X_test, y_test = form_Xy(test)
df_test = make_lama_df(X_test, y_test)

roles = {
    "target": "label",
    "group": "cluster"
}

task = Task("binary")

automl = TabularAutoML(
    task=task,
    reader_params={'random_state': SEED})

oof_pred = automl.fit_predict(
    df_train,
    roles=roles)

valid_prob = oof_pred.data[:, 0]
valid_pred = (valid_prob > 0.5) * 1
metrics = Metrics(y_train, valid_pred, valid_prob, "valid")
valid_metrics = metrics.get_metrics()
logger.report_table(title='Validation metrics', series='pandas DataFrame',
                    table_plot=valid_metrics)


test_pred = automl.predict(df_test)
test_prob = test_pred.data.reshape(-1, )
test_pred = (test_pred.data[:, 0] > 0.5) * 1
metrics = Metrics(y_test, test_pred, test_prob,
                  test_data + f"identity_{identity}")
test_metrics = metrics.get_metrics()

logger.report_table(title='Test metrics', series='pandas DataFrame',
                    table_plot=test_metrics)


plot_roc_curve(y_test, test_prob, test_data)
