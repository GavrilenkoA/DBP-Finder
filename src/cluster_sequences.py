import pandas as pd
from utils import write_fasta, add_clusters, exclude_common_train_seqs
import subprocess
import os


def cluster_data(train_csv: str, test_csv: str, identity: float = 0.5) -> None:

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train = exclude_common_train_seqs(train, test)
    df = pd.concat([train, test])

    # Prepare fasta before clustering
    write_fasta(df, "merged.fasta")

    # Clustering
    fasta_input = "../data/fasta/merged.fasta"
    output_dir = "../data/clusters/merged"

    coverage = identity + 0.1

    subprocess.run(f"mmseqs easy-cluster {fasta_input} {output_dir} tmp --min-seq-id {identity} -c {coverage} --cov-mode 0", shell=True)

    # Parse clusters
    output_mmseqs = pd.read_csv("../data/clusters/merged_cluster.tsv", sep="\t", header=None)
    output_mmseqs = add_clusters(output_mmseqs)
    assert len(output_mmseqs) == len(df), f"{len(output_mmseqs)}, {len(df)}"

    test_name = os.path.basename(test_csv).split(".csv")[0]
    output_mmseqs.to_csv(f"../data/ready_data/{test_name}_train_{identity}.csv", index=False)


def main():
    test_data = input()
    train_csv = "../data/embeddings/input_csv/train_p2.csv"
    test_csv = f"../data/embeddings/input_csv/{test_data}.csv"
    identity = 0.5

    cluster_data(train_csv, test_csv, identity=identity)


main()
