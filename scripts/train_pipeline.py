import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import clearml
from clearml import Task as clearml_task
from clearml import Logger

from scripts.utils import SEED
from scripts.train_src import (make_lama_df, merge_embed, Metrics, form_Xy, plot_roc_curve)


input_test = input()

clearml.browser_login()
task_ml = clearml_task.init(
    project_name="DBPs_search",
    task_name=f"{input_test}",
    output_uri=True
)
logger = Logger.current_logger()

train = pd.read_csv(f"data/ready_data/train_{input_test}.csv")
test = pd.read_csv(f"data/embeddings/input_csv/{input_test}.csv")


train = merge_embed(train, "train_p2")
X_train, y_train, clusters_train = form_Xy(train, clusters="Yes")
df_train = make_lama_df(X_train, y_train, clusters_train)

test = merge_embed(test, input_test)
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
                  input_test)
test_metrics = metrics.get_metrics()

logger.report_table(title='Test metrics', series='pandas DataFrame',
                    table_plot=test_metrics)


plot_roc_curve(y_test, test_prob, input_test)
