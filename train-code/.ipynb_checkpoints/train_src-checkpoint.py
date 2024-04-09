import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, matthews_corrcoef, roc_auc_score, f1_score)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SEED = 42


def make_balanced_df(df, seed=SEED):
    pos_cls = df[df.label == 1]
    neg_cls = df[df.label == 0]
    if len(neg_cls) > len(pos_cls):
        neg_cls = neg_cls.sample(n=len(pos_cls), random_state=seed)
    elif len(neg_cls) < len(pos_cls):
        pos_cls = pos_cls.sample(n=len(neg_cls), random_state=seed)
    balanced_df = pd.concat([pos_cls, neg_cls])
    balanced_df = balanced_df.sample(frac=1, random_state=seed)
    return balanced_df


def reduce_train(clusters_train_test, train, test):
    a = clusters_train_test.merge(train, on=["identifier"])
    b = clusters_train_test.merge(test, on=["identifier"])
    
    exclude_train = a.merge(b, on=["cluster"])["identifier_x"].drop_duplicates()
    train = train.loc[~train["identifier"].isin(exclude_train)]
    train = make_balanced_df(train)
    train = train.merge(clusters_train_test, on=["identifier"])
    return train


def load_obj(file_path: str):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def get_embeds(file_path: str) -> pd.DataFrame:
    dict_data = load_obj(file_path)
    df = pd.DataFrame(list(dict_data.items()), columns=["identifier", "embedding"])
    return df


def merge_embed(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    embed_df = get_embeds(file_path)
    out_df = df.merge(embed_df, on="identifier")
    assert len(out_df) == len(df)
    return out_df


def make_lama_df(X, y, clusters=None) -> pd.DataFrame:
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
    X = process_embeddings(df['embedding'])
    y = df['label'].values
    if clusters is not None:
        clusters = df['cluster'].values
        return X, y, clusters
    return X, y


def make_inference_lama_df(df: pd.DataFrame) -> pd.DataFrame:
    X = process_embeddings(df['embedding'])
    df = pd.DataFrame(X)
    df.columns = [f"component_{i}" for i in range(df.shape[1])]
    return df


class Metrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, descr: str) -> None:
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
            'accuracy': accuracy_val,
            'sensitivity': recall_val,
            'specificity': specificity_val,
            'precision': precision_val,
            'AUC': roc_auc_value,
            'F1': f1_value,
            'MCC': mcc_val
        }

        metrics_df = pd.DataFrame(metrics_dict, index=[self.descr])
        return metrics_df


def plot_roc_curve(y_true, y_pred, name_dataset):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve on {name_dataset} dataset')
    plt.legend(loc="lower right")
    plt.show()