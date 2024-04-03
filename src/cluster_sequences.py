import pandas as pd
from utils import write_fasta, add_clusters, exclude_common_train_seqs
import subprocess


identity = 0.15
coverage = identity + 0.1

train = pd.read_csv("../data/embeddings/input_csv/train_p2.csv")
test = pd.read_csv("../data/embeddings/input_csv/trembl_test.csv")

train = exclude_common_train_seqs(train, test)

df = pd.concat([train, test])

# Prepare fasta before clustering
write_fasta(df, "smth.fasta")

# Clustering
fasta_input = f"../data/fasta/smth.fasta"
output_dir = f"../data/clusters/smth"

subprocess.run(f"mmseqs easy-cluster {fasta_input} {output_dir} tmp --min-seq-id {identity} -c {coverage} --cov-mode 0", shell=True)


# Prepare df with clusters before merge
output_mmseqs = pd.read_csv("../data/clusters/smth_cluster.tsv", sep="\t", header=None)
output_mmseqs = add_clusters(output_mmseqs)
output_mmseqs = output_mmseqs.loc[:, ["identifier", "cluster"]]

assert len(output_mmseqs) == len(df), f"{len(output_mmseqs)}, {len(df)}"

output_mmseqs.to_csv(f"../data/ready_data/trembl_test_train_{identity}.csv", index=False)