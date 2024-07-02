import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def load_obj(file_path: str) -> dict[str, np.ndarray]:
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def get_embeds(file_path: str) -> pd.DataFrame:
    dict_data = load_obj(file_path)
    df = pd.DataFrame(list(dict_data.items()), columns=["identifier", "embedding"])
    return df


def merge_embed(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    file_path = f"data/embeddings/ankh_embeddings/{file_path}.pkl"
    embed_df = get_embeds(file_path)
    out_df = df.merge(embed_df, on="identifier")
    assert len(out_df) == len(df)
    return out_df


def make_lama_df(
    X: np.ndarray, y: np.ndarray, clusters: None | str = None
) -> pd.DataFrame:
    df = pd.DataFrame(X)
    df.columns = [f"component_{i}" for i in range(df.shape[1])]
    df["label"] = y
    if clusters is not None:
        df["cluster"] = clusters
    return df


def process_embeddings(embeddings: pd.Series) -> np.ndarray:
    embeddings = [item.reshape(1, -1) for item in embeddings.tolist()]
    embeddings = np.concatenate(embeddings)
    return embeddings


def form_Xy(df: pd.DataFrame, clusters: None | str = None):
    X = process_embeddings(df["embedding"])
    y = df["label"].values
    if clusters is not None:
        clusters = df["cluster"].values
        return X, y, clusters
    return X, y


def make_inference_lama_df(df: pd.DataFrame) -> pd.DataFrame:
    X = process_embeddings(df["embedding"])
    df = pd.DataFrame(X)
    df.columns = [f"component_{i}" for i in range(df.shape[1])]
    return df


class Metrics:
    def __init__(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, descr: str
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.descr = descr

    def get_metrics(self) -> pd.DataFrame:
        accuracy_val = accuracy_score(self.y_true, self.y_pred)
        recall_val = recall_score(self.y_true, self.y_pred)
        precision_val = precision_score(self.y_true, self.y_pred)
        specificity_val = recall_score(self.y_true, self.y_pred, pos_label=0)
        mcc_val = matthews_corrcoef(self.y_true, self.y_pred)
        f1_value = f1_score(self.y_true, self.y_pred)
        roc_auc_value = roc_auc_score(self.y_true, self.y_prob)

        metrics_dict = {
            "accuracy": accuracy_val,
            "sensitivity": recall_val,
            "specificity": specificity_val,
            "precision": precision_val,
            "AUC": roc_auc_value,
            "F1": f1_value,
            "MCC": mcc_val,
        }

        metrics_df = pd.DataFrame(metrics_dict, index=[self.descr])
        return metrics_df


def plot_roc_curve(y_true, y_pred, name_dataset):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, label="ROC Curve (area = %0.2f)" % auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve on {name_dataset} dataset")
    plt.legend(loc="lower right")
    plt.show()


def predict(df_test):
    model = joblib.load("models/DBP-finder.pkl")
    test_pred = model.predict(df_test)
    test_prob = test_pred.data.reshape(
        -1,
    )
    test_pred = (test_pred.data[:, 0] > 0.5) * 1
    return test_prob, test_pred


def filter_test_by_kingdom(test: pd.DataFrame, test_input: str, kingdom: str):
    df = pd.read_csv(f"data/processed/{test_input}_kingdom.csv")
    subset_df = df[df["kingdom"] == f"{kingdom}"]
    test = test.merge(subset_df, on="identifier")
    return test
