import pandas as pd
import requests
import logging
import os


def main():
    input_csv = input()

    output_file = os.path.splitext(os.path.basename(input_csv))[0]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=f"{output_file}.log",
    )

    df = pd.read_csv(input_csv)

    identifiers = []
    sequences = []

    for i, chain in enumerate(df["identifier"].tolist()):
        id_ = chain[:-1]
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{id_}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            pdb_id = list(data.keys())[0]
            uniprot_id = list(data[pdb_id]["UniProt"].keys())[0]

            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            response = requests.get(url)
            if response.status_code == 200:
                fasta_data = response.text
                seq = "".join(fasta_data.split("\n")[1:])
                identifiers.append(uniprot_id)
                sequences.append(seq)
            else:
                logging.info(f"{uniprot_id} is not a valid UniProt")
        else:
            logging.info(f"{id_} doesn't map RCSB id to UniProt id")

    df = pd.DataFrame({'identifier': identifiers, 'sequence': sequences})
    df.to_csv(output_file + ".csv", index=False)


if __name__ == '__main__':
    main()
