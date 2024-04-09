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


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df.loc[df["sequence"].apply(valid_sequence)]
    df = df.loc[df["sequence"].apply(lambda x: 49 < len(x) < 1025)]
    df = df.drop_duplicates(subset=["sequence"])
    return df


def add_source(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df["identifier"] = df["identifier"].apply(lambda x: x + f"+{name}")
    return df


def exclude_same_seqs(train, test):
    common_identifiers = test.merge(train, on=["sequence"])["identifier_x"]
    test = test.loc[~test["identifier"].isin(common_identifiers)]
    return test


def write_fasta(df: pd.DataFrame, name_file: str) -> None:
    def pull_data(x):
        id_ = x["identifier"]
        seq = x["sequence"]
        return id_, seq

    data = df.apply(lambda x: pull_data(x), axis=1).tolist()

    with open(f"../data/fasta/{name_file}", "w") as file:
        for item in data:
            file.write(">" + f"{item[0]}")
            file.write("\n")
            file.write(f"{item[1]}")
            file.write("\n")


# def filter_train(clusters_train_test, train, test):
#     a = clusters_train_test.merge(train, on=["identifier"])
#     b = clusters_train_test.merge(test, on=["identifier"])

#     exclude_train = a.merge(b, on=["cluster"])["identifier_x"].drop_duplicates()
#     train = train.loc[~train["identifier"].isin(exclude_train)]
#     train = clusters_train_test.merge(train, on=["identifier"])
#     return train


# def filter_test(clusters_train_test, train, test):
#     a = clusters_train_test.merge(train, on=["identifier"])
#     b = clusters_train_test.merge(test, on=["identifier"])

#     exclude_test = a.merge(b, on=["cluster"])["identifier_y"].drop_duplicates()
#     filtered_test = test.loc[~test["identifier"].isin(exclude_test)]
#     return filtered_test
