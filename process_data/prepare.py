# from get_not_annotated_proteins import collect_data, filter_df
# import os
# import pandas as pd


# input_fasta = "../data/uniprot/Human_dnabinders_4.fasta"
# name_file, _ = os.path.splitext(os.path.basename(input_fasta))

# identifiers, sequences = collect_data(input_fasta)
# df = pd.DataFrame({"identifier": identifiers, "sequence": sequences})
# df = filter_df(df)
# df.to_csv(f"../data/embeddings/input_csv/{name_file}.csv", index=False)
