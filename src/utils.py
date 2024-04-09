import pandas as pd


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


def add_clusters(df: pd.DataFrame):
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
    assert not pd.concat([train, test])["identifier"].duplicated().any()  # concatenated df should not contain seqs
    # with the same id
    common_id = test.merge(train, on=["sequence"])["identifier_y"]
    train = train.loc[~train["identifier"].isin(common_id)]
    return train
