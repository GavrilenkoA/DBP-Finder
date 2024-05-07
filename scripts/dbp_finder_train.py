import pandas as pd
import joblib
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from train_src import make_lama_df, merge_embed, form_Xy
from utils import SEED

train = pd.read_csv("data/ready_data/my_train0.5.csv")

train = merge_embed(train, "data/embeddings/ankh_embeddings/train_p2.pkl")
X_train, y_train, clusters_train = form_Xy(train, clusters="Yes")
df_train = make_lama_df(X_train, y_train, clusters=clusters_train)

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


joblib.dump(automl, "models/DBP-finder.pkl")
