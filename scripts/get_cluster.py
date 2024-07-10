import argparse

import pandas as pd
from utils import add_clusters, cluster_sequences, write_fasta


def cluster_data(path_data: str) -> pd.DataFrame:
    df = pd.read_csv(path_data)

    write_fasta(df, "merged.fasta")

    output_mmseqs = cluster_sequences()
    output_mmseqs = add_clusters(output_mmseqs)
    clustered_df = df.merge(output_mmseqs, on="identifier")
    assert len(clustered_df) == len(df), f"{len(clustered_df)}, {len(df)}"
    return clustered_df


def main():
    parser = argparse.ArgumentParser(
        description="Program\
                                    for clustering protein sequences"
    )
    parser.add_argument("input_path", type=str, help="A string argument")
    parser.add_argument("output_path", type=str, help="A string argument")

    args = parser.parse_args()
    df = cluster_data(args.input_path)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
