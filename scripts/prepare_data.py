import pandas as pd
from cluster_sequences import cluster_data
from utils import reduce_train, make_balanced_df

input_test = input()

train = pd.read_csv("data/embeddings/input_csv/train_p2.csv")
test = pd.read_csv(f"data/embeddings/input_csv/{input_test}.csv")


output_mmseq = cluster_data(train, test, identity=0.5)
output_mmseq = reduce_train(output_mmseq)

train["identifier"] = train.identifier.apply(lambda x: x.split("_")[0])  # fix later
train = train.merge(output_mmseq, on="identifier")
train = make_balanced_df(train)

train.to_csv(f"data/ready_data/train_{input_test}.csv", index=False)
