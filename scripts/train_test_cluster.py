import pandas as pd
from utils import (add_clusters, add_source_to_id, cluster_sequences,
                   delete_source_from_id, exclude_common_train_seqs,
                   write_fasta)


def cluster_data(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:

    train, test = add_source_to_id(train, test)
    train = exclude_common_train_seqs(train, test)

    df = pd.concat([train, test])

    # Prepare fasta before clustering
    write_fasta(df, "merged.fasta")

    # Clustering
    output_mmseqs = cluster_sequences()

    output_mmseqs = add_clusters(output_mmseqs)
    assert len(output_mmseqs) == len(df), f"{len(output_mmseqs)}, {len(df)}"

    output_mmseqs = delete_source_from_id(output_mmseqs)

    return output_mmseqs
