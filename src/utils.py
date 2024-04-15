import pandas as pd


def make_balanced_df(df, seed=42):
    pos_cls = df[df.label == 1]
    neg_cls = df[df.label == 0]
    if len(neg_cls) > len(pos_cls):
        neg_cls = neg_cls.sample(n=len(pos_cls), random_state=seed)
    elif len(neg_cls) < len(pos_cls):
        pos_cls = pos_cls.sample(n=len(neg_cls), random_state=seed)
    balanced_df = pd.concat([pos_cls, neg_cls])
    return balanced_df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df.loc[df["sequence"].apply(valid_sequence)]
    df = df.loc[df["sequence"].apply(lambda x: 49 < len(x) < 1025)]
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


def convert_fasta_to_df(fasta_file: str, mode: int = 1, class_: int = 0) -> pd.DataFrame:
    with open(fasta_file) as fi:
        data = fi.readlines()

    identifiers = []
    sequences = []
    seq = ""

    for chunk in data:
        if chunk.startswith(">"):
            head = chunk.split("|")[mode].replace(">", "")
            identifiers.append(head)
            if seq:
                sequences.append(seq)
                seq = ""
        else:
            seq += chunk.strip()

    if seq:
        sequences.append(seq)

    labels = [class_] * len(identifiers)

    df = pd.DataFrame({"identifier": identifiers, "label": labels,
                       "sequence": sequences})
    return df


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
    df = df.loc[:, ["identifier", "cluster"]]
    return df


def exclude_common_train_seqs(train, test):
    common_id = train.merge(test, on=["identifier"])["identifier"]  # delete common ids if they exists
    train = train.loc[~train["identifier"].isin(common_id)]

    common_id = test.merge(train, on=["sequence"])["identifier_y"] # delete common seqs from train
    train = train.loc[~train["identifier"].isin(common_id)]
    return train


def add_source_to_id(train: pd.DataFrame, test: pd.DataFrame, test_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train["identifier"] = train["identifier"].apply(lambda x: x + "_train")
    test["identifier"] = test["identifier"].apply(lambda x: x + f"_{test_name}")
    return train, test


def delete_source_from_id(df: pd.DataFrame) -> pd.DataFrame:
    df["source"] = df.identifier.apply(lambda x: x.split("_")[1])
    df["identifier"] = df.identifier.apply(lambda x: x.split("_")[0])
    return df


def delete_common_seqs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_seqs = test.merge(train, on=["sequence"])["sequence"]  # delete common seqs from both dataframes

    train_sep = train.loc[~train["sequence"].isin(common_seqs)]
    test_sep = test.loc[~test["sequence"].isin(common_seqs)]

    return train_sep, test_sep


def find_common_seqs(train: pd.DataFrame, test: pd.DataFrame) -> list:
    common_seqs = test.merge(train, on=["sequence"])["identifier_x"].to_list()  # find common seqs in test
    return common_seqs


def intersect_cluster_seq(df: pd.DataFrame, test_name: str) -> list:
    cluster_seqs = []
    grouped = df.groupby("cluster")
    for _, group in grouped:
        if group["source"].nunique() > 1:
            id_ = group[group["source"] == test_name]["identifier"].to_list()
            cluster_seqs.extend(id_)
    return cluster_seqs


def assign_group_cluster(basename: str) -> pd.DataFrame:
    id_0_25 = pd.read_csv(f"data/{basename}_0.25.csv")
    id_0_5 = pd.read_csv(f"data/{basename}_0.5.csv")
    id_0_75 = pd.read_csv(f"data/{basename}_0.75.csv")
    id_0_1_group = pd.read_csv(f"data/{basename}_1.csv")

    target_df = pd.read_csv(f"data/embeddings/input_csv/{basename}.csv")

    p_1 = pd.concat([id_0_5, id_0_75, id_0_1_group]).drop_duplicates()
    p_2 = pd.concat([id_0_75, id_0_1_group]).drop_duplicates()
    assert not id_0_1_group["identifier"].duplicated().any(), "Duplicates found within the identifier column"

    id_0_25_group = id_0_25.loc[~id_0_25["identifier"].isin(p_1["identifier"])]
    id_0_5_group = id_0_5.loc[~id_0_5["identifier"].isin(p_2["identifier"])]
    id_0_75_group = id_0_75.loc[~id_0_75["identifier"].isin(id_0_1_group["identifier"])]

    id_0_25_group.loc[:, "group"] = ["id_0.25"] * len(id_0_25_group)
    id_0_5_group.loc[:, "group"] = ["id_0.5"] * len(id_0_5_group)
    id_0_75_group.loc[:, "group"] = ["id_0.75"] * len(id_0_75_group)
    id_0_1_group.loc[:, "group"] = ["id_1"] * len(id_0_1_group)
    group_data = pd.concat([id_0_25_group, id_0_5_group, id_0_75_group, id_0_1_group])
    target_df = target_df.merge(group_data, on="identifier", how="left")

    idx_na = target_df.group.isna()
    target_df = target_df.loc[idx_na, "group"] = "<threshold"
    return target_df
