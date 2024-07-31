import re
import subprocess
from functools import wraps

import h5py
import pandas as pd

SEED = 42


def make_balanced_df(df, seed=SEED):
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


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df[df["sequence"].apply(valid_sequence)]
    df = df[df["sequence"].apply(lambda x: 49 < len(x) < 1025)]
    df = df.drop_duplicates(subset=["sequence"])
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


def collect_df(content: str) -> pd.DataFrame:
    lines = content.split("\n")
    identifiers = []
    sequences = []
    seq = ""

    for line in lines:
        if line.startswith(">"):
            head = line.split("|")[1]
            identifiers.append(head)
            if seq:
                sequences.append(seq)
                seq = ""
        else:
            seq += line.strip()

    if seq:
        sequences.append(seq)

    assert len(identifiers) == len(sequences)
    df = pd.DataFrame({"identifier": identifiers, "sequence": sequences})
    return df


def convert_fasta_to_df(fasta_file: str) -> pd.DataFrame:
    with open(fasta_file) as fi:
        content = fi.read()
    df = collect_df(content)
    return df


def extract_columns(
    fasta_file: str, column1: str = "identifier", column2: str = "sequence"
) -> tuple[list]:
    df = convert_fasta_to_df(fasta_file)
    part_1 = df.loc[:, column1].to_list()
    part_2 = df.loc[:, column2].to_list()
    return part_1, part_2


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


def find_common_seqs(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    # find common seqs in test
    common_seqs = test.merge(train, on=["sequence"])["identifier_x"].to_list()
    return common_seqs


def save_dict_to_hdf5(data_dict, filename):
    """
    Save a dictionary with string keys and NumPy array values to an HDF5 file.

    Parameters:
    data_dict (dict): Dictionary with string keys and NumPy array values.
    filename (str): Name of the HDF5 file to save the data.
    """
    with h5py.File(filename, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)


def load_dict_from_hdf5(filename):
    """
    Load a dictionary with string keys and NumPy array values from an HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file to load the data from.

    Returns:
    dict: Dictionary with string keys and NumPy array values.
    """
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


class RNADataset:
    def __init__(self, path: str) -> None:
        self.path = path

    def raname_columns(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df.rename(columns={"Meta": "identifier", "pep": "sequence"}, inplace=True)
        return df

    def extract_test_counts(self) -> tuple[int, int]:
        # Compile the regex pattern to find number after 'TeP'
        pattern_pos = re.compile(r".*TeP(\d+).*")
        pattern_neg = re.compile(r".*TeN(\d+).*")

        # Perform the match
        match_1 = pattern_pos.match(self.path)
        match_2 = pattern_neg.match(self.path)

        if match_1 and match_2:
            return int(match_1.group(1)), int(match_2.group(1))
        else:
            raise ValueError("Input string does not match expected pattern.")

    @staticmethod
    def filter_length(df: pd.DataFrame, threshold: int = 1024):
        df["sequence"] = df["sequence"].apply(
            lambda seq: seq[:threshold] if len(seq) > threshold else seq
        )

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.raname_columns()
        counts_pos, counts_neg = self.extract_test_counts()

        pos_samples = df[df["label"] == 1].tail(counts_pos)
        neg_samples = df[df["label"] == 0].tail(counts_neg)
        sample = pd.concat([pos_samples, neg_samples])
        self.filter_length(sample)
        return sample
