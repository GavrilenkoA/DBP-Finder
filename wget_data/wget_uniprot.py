import pandas as pd
import requests
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"uniprot_parsing.log",
)


def wget_fasta() -> None:
    null_seqs_df = pd.read_csv(path_csv)
    null_seqs_df.index = np.arange(len(null_seqs_df))

    sequences = []
    heads = []
    class_binders = []
    for i, id_ in enumerate(null_seqs_df["identifier"].to_list()):
        url = f"https://rest.uniprot.org/uniprotkb/{id_}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            fasta_data = response.text
            seq = "".join(fasta_data.split("\n")[1:])

            heads.append(id_)
            sequences.append(seq)
            class_binders.append(null_seqs_df.iloc[i]["class_binder"])

            logging.info(f"{id_} - processed")

    df = pd.DataFrame({"identifier": heads, "class_binder": class_binders, "sequence": sequences})
    df.to_csv(path_csv, index=False)


if __name__ == "__main__":
    path_csv = input()
    wget_fasta()
