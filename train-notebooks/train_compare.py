import pandas as pd
import joblib
from train_src import make_lama_df, merge_embed, Metrics, form_Xy

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

import clearml

clearml.browser_login()

from clearml import Task as clearml_task
from clearml import Logger

SEED = 42
TIMEOUT = 3600

test_name = input()
model_name = input()

task_ml = clearml_task.init(
    project_name="DBPs_search",
    task_name=f"{test_name}_{model_name}",
    output_uri=False
)

logger = Logger.current_logger()

train = pd.read_csv("../data/embeddings/input_csv/train_p2.csv")
train = train.sample(frac=1, random_state=SEED)
test = pd.read_csv(f"../data/embeddings/input_csv/{test_name}.csv")


train = merge_embed(train, f"../data/embeddings/{model_name}_embeddings/train_p2.pkl")
X_train, y_train = form_Xy(train)
df_train = make_lama_df(X_train, y_train)

test = merge_embed(test, f"../data/embeddings/{model_name}_embeddings/{test_name}.pkl")
X_test, y_test = form_Xy(test)
df_test = make_lama_df(X_test, y_test)


roles = {
    "target": "label"
}

task = Task("binary")

automl = TabularAutoML(
    task=task,
    timeout=TIMEOUT,
    reader_params={'random_state': SEED})


oof_pred = automl.fit_predict(
    df_train,
    roles=roles)


#valid_prob = oof_pred.data[:, 0]
#valid_pred = (valid_prob > 0.5) * 1
#metrics = Metrics(y_train, valid_pred, valid_prob, "valid")
#valid_metrics = metrics.get_metrics()
#logger.report_table(title='Validation metrics', series='pandas DataFrame', table_plot=valid_metrics)


test_pred = automl.predict(df_test)
test_prob = test_pred.data.reshape(-1, )
test_pred = (test_pred.data[:, 0] > 0.5) * 1
metrics = Metrics(y_test, test_pred, test_prob, test_name)
test_metrics = metrics.get_metrics()
logger.report_table(title='Test metrics', series='pandas DataFrame', table_plot=test_metrics)

joblib.dump(automl, f"models/{model_name}_{test_name}.pkl")
