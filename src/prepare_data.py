import pandas as pd
from src.utils import convert_fasta_to_df, filter_df


binders = convert_fasta_to_df("data/uniprot/go_0003677_swissprot.fasta")
binders.loc[:, "label"] = 1

non_binders = convert_fasta_to_df("data/uniprot/notgo_0003723_notgo_0003677_swissprot.fasta")
non_binders.loc[:, "label"] = 0
train = pd.concat([binders, non_binders])
train = filter_df(train)
train.to_csv("data/embeddings/input_csv/train_p2.csv", index=False)
