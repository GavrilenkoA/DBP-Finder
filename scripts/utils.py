import pickle
import hashlib
import subprocess
from functools import wraps
from typing import Generator, TextIO, Tuple

import h5py
import pandas as pd
import yaml


def prepare_neg_samples(path: str) -> pd.DataFrame:
    with open(path, "r") as file:
        neg_samples_annotation = yaml.safe_load(file)

    identifiers = list()
    for protein_id in neg_samples_annotation:
        if not neg_samples_annotation[protein_id]:
            identifiers.append(protein_id)

    df = pd.DataFrame(list(identifiers), columns=["identifier"])
    return df


def make_balanced_df(df, seed=42):
    pos_cls = df[df.label == 1]
    neg_cls = df[df.label == 0]
    if len(neg_cls) > len(pos_cls):
        neg_cls = neg_cls.sample(n=len(pos_cls), random_state=seed)
    elif len(neg_cls) < len(pos_cls):
        pos_cls = pos_cls.sample(n=len(neg_cls), random_state=seed)
    balanced_df = pd.concat([pos_cls, neg_cls])
    return balanced_df


def reduce_train(output_mmseq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set = output_mmseq[output_mmseq["source"] == "train"]
    test_set = output_mmseq[output_mmseq["source"] == "test"]

    exclude_train = train_set.merge(test_set, on=["cluster"])[
        "identifier_x"
    ].drop_duplicates()
    train_set = train_set[~train_set["identifier"].isin(exclude_train)]

    intersect_clusters_df = train_set.merge(test_set, on="cluster")
    assert len(intersect_clusters_df) == 0, "Train and test intersect clusters"

    train_set = train_set.drop("source", axis=1)
    test_set = test_set.drop("source", axis=1)
    return train_set, test_set


def save_csv(
    df: pd.DataFrame, basename: str, path: str = "data/embeddings/input_csv/"
) -> None:
    path = path + f"{basename}.csv"
    df.to_csv(path, index=False)


def filter_df(df: pd.DataFrame, min_len: int = 50, max_len: int = 1024) -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df[df["sequence"].apply(valid_sequence)]
    df = df[df["sequence"].apply(lambda x: min_len <= len(x) <= max_len)]
    df = df.drop_duplicates(subset=["sequence"])
    df = df.drop_duplicates(subset=["identifier"])
    return df


def write_fasta(df: pd.DataFrame, name_file: str) -> None:
    def pull_data(x):
        id_ = x["identifier"]
        seq = x["sequence"]
        return id_, seq

    data = df.apply(lambda x: pull_data(x), axis=1).tolist()

    with open(f"data/fasta/{name_file}", "w") as file:
        for item in data:
            file.write(">" + f"{item[0]}")
            file.write("\n")
            file.write(f"{item[1]}")
            file.write("\n")


def read_fasta(f: TextIO) -> Generator[Tuple[str, str], None, None]:
    for line in f:
        if line.startswith(">"):
            name = line.strip()[1:]
            break
    seqs = []
    for line in f:
        if line.startswith(">"):
            yield name, "".join(seqs)
            seqs = []
            name = line.strip()[1:]
        else:
            seqs.append(line.rstrip())
    yield name, "".join(seqs)


def convert_fasta_to_df(fasta_file: str) -> pd.DataFrame:
    with open(fasta_file) as fi:
        df = pd.DataFrame(read_fasta(fi))
    df.columns = ["identifier", "sequence"]
    return df


def select_columns(columns):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            return df.loc[:, columns]

        return wrapper

    return decorator


@select_columns(["identifier", "cluster"])
def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["repr", "identifier"]

    reprs = df["repr"].to_list()
    clusters = []
    count = 0
    name = ""
    for item in reprs:
        if item != name:
            count += 1
            name = item
        clusters.append(count)

    df["cluster"] = clusters
    return df


def assert_no_duplicates(df: pd.DataFrame, column: str = "identifier") -> None:
    assert not df[column].duplicated().any(), f"Column '{column}' contains duplicates identifiers"


def exclude_common_train_seqs(train, test):
    # delete common seqs from train
    common_id = train.merge(test, on=["sequence"])["identifier_x"]
    train = train[~train["identifier"].isin(common_id)]
    return train


def add_source_to_id(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame]:
    train_ = train.copy(deep=True)
    test_ = test.copy(deep=True)

    train_["identifier"] = train_["identifier"].apply(lambda x: x + "_train")
    test_["identifier"] = test_["identifier"].apply(lambda x: x + "_test")
    return train_, test_


def delete_source_from_id(df: pd.DataFrame) -> pd.DataFrame:
    df["source"] = df.identifier.apply(lambda x: x.split("_")[1])
    df["identifier"] = df.identifier.apply(lambda x: x.split("_")[0])
    return df


def delete_common_seqs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame]:
    # delete common seqs from both dataframes
    common_seqs = test.merge(train, on=["sequence"])["sequence"]
    train = train[~train["sequence"].isin(common_seqs)]
    test = test[~test["sequence"].isin(common_seqs)]
    return train, test


def save_embeds(obj, data_name: str, model_name: str):
    filename = f"data/embeddings/{model_name}_embeddings/{data_name}.pkl"
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def save_dict_to_hdf5(data_dict, filename):
    with h5py.File(filename, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)


def load_dict_from_hdf5(filename):
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def cluster_sequences(
    fasta_input: str = "data/fasta/merged.fasta",
    output_dir: str = "data/clusters/merged",
    identity: float = 0.5,
) -> pd.DataFrame:
    # Calculate coverage
    coverage = identity + 0.1

    # Run mmseqs easy-cluster
    subprocess.run(
        f"mmseqs easy-cluster {fasta_input} {output_dir} tmp --min-seq-id {identity} -c {coverage} --cov-mode 0",
        shell=True,
        check=True,  # Raises an error if the command fails
    )

    # Parse clusters
    output_mmseqs = pd.read_csv(f"{output_dir}_cluster.tsv", sep="\t", header=None)
    return output_mmseqs


def split_with_overlap(sequence, chunk_size=1024, overlap=512):
    chunks = []
    length = len(sequence)

    for start in range(0, length, chunk_size - overlap):
        end = start + chunk_size
        chunks.append(sequence[start:end])
    return chunks


def chunk_dataframe(df, chunk_size=1024, overlap=512):
    # Создаем списки для новых данных
    new_identifiers = []
    new_sequences = []

    # Проходим по каждой строке исходного датафрейма
    for _, row in df.iterrows():
        identifier = row['identifier']
        sequence = row['sequence']

        # Разбиваем последовательность на чанки
        chunks = split_with_overlap(sequence, chunk_size, overlap)

        # Добавляем чанки в новые списки с нумерацией
        for i, chunk in enumerate(chunks, 1):
            new_identifiers.append(f"{identifier}_{i}")
            new_sequences.append(chunk)

    # Создаем новый датафрейм
    chunked_df = pd.DataFrame({
        'identifier': new_identifiers,
        'sequence': new_sequences
    })

    return chunked_df


def postprocess(df):
    # Извлекаем префикс идентификатора (без номера чанка)
    df["prefix"] = df["identifier"].str.extract(r"^(.*?)_\d+$")

    # Если в identifier не было номера чанка (например, исходные данные),
    # то используем весь identifier как prefix
    df["prefix"] = df["prefix"].fillna(df["identifier"])

    # Группируем по префиксу и находим индекс строки с максимальным скором в каждой группе
    idx = df.groupby("prefix")["probability"].idxmax()

    # Выбираем соответствующие строки
    result_df = df.loc[idx].reset_index(drop=True)

    # Удаляем временную колонку identifier и переименовываем prefix в identifier
    result_df = result_df.drop(columns="identifier").rename(columns={"prefix": "identifier"})

    return result_df


def hash_suffix(s, length=10):
    return hashlib.md5(s.encode()).hexdigest()[:length]
