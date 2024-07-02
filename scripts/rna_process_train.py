import pandas as pd
from utils import convert_fasta_to_df, filter_df, make_balanced_df

binders = convert_fasta_to_df(
    "data/rna/raw/uniprotkb_AND_go_0003723_AND_reviewed_t_2024_06_23.fasta"
)
binders["label"] = 1

non_binders_1 = pd.read_csv("data/rna/processed/4_annot_score_neg_cl.csv")
non_binders_2 = pd.read_csv("data/rna/processed/5_annot_score_neg_cl.csv")
non_binders = pd.concat([non_binders_1, non_binders_2])
non_binders["label"] = 0

train = pd.concat([binders, non_binders])
train = filter_df(train)
train = make_balanced_df(train)
train.to_csv("data/rna/processed/train.csv", index=False)
