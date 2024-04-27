import argparse
import pandas as pd
from utils import (write_fasta, add_clusters, find_common_seqs,
                   add_source_to_id, delete_common_seqs,
                   delete_source_from_id, intersect_cluster_seq)
import subprocess
import os


def cluster_data(train_csv: str, test_csv: str, identity: float = 0.5) -> None:

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    common_seqs = find_common_seqs(train, test)
    train, test = delete_common_seqs(train, test)

    test_name = os.path.basename(test_csv).split(".csv")[0]

    train, test = add_source_to_id(train, test, test_name)

    df = pd.concat([train, test])

    # Prepare fasta before clustering
    write_fasta(df, "merged.fasta")

    # Clustering
    fasta_input = "data/fasta/merged.fasta"
    output_dir = "data/clusters/merged"

    coverage = identity + 0.1

    subprocess.run(f"mmseqs easy-cluster {fasta_input} {output_dir}\
                   tmp --min-seq-id {identity} -c {coverage} --cov-mode 0",
                   shell=True)

    # Parse clusters
    output_mmseqs = pd.read_csv("data/clusters/merged_cluster.tsv",
                                sep="\t", header=None)
    output_mmseqs = add_clusters(output_mmseqs)
    assert len(output_mmseqs) == len(df), f"{len(output_mmseqs)}, {len(df)}"

    output_mmseqs = delete_source_from_id(output_mmseqs)
    cluster_seqs = intersect_cluster_seq(output_mmseqs, test_name)

    common_seqs = pd.DataFrame({"identifier": common_seqs})
    cluster_seqs = pd.DataFrame({"identifier": cluster_seqs})

    cluster_seqs.to_csv(f"data/not_annotated/clustered_data/\
                        {test_name}_{identity}.csv", index=False)
    common_seqs.to_csv(f"data/not_annotated/clustered_data/\
                       {test_name}_1.csv", index=False)


def main(test_data, identity):
    train_csv = "data/embeddings/input_csv/train_p2.csv"
    test_csv = f"data/embeddings/input_csv/{test_data}.csv"
    cluster_data(train_csv, test_csv, identity=identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program\
                                    for clustering protein sequences")
    parser.add_argument("test_data", type=str, help="A string argument")
    parser.add_argument("identity_value", type=float, help="A float argument")
    args = parser.parse_args()
    main(args.test_data, args.identity_value)
