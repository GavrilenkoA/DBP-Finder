import argparse
import pandas as pd
import os


def write_train_fasta(df: pd.DataFrame, name_file: str) -> None:
    def pull_data(x):
        id_ = x["identifier"]
        seq = x["sequence"]
        return id_, seq

    data = df.apply(lambda x: pull_data(x), axis=1).tolist()

    with open(name_file, "w") as file:
        for item in data:
            file.write(">" + f"{item[0]}")
            file.write("\n")
            file.write(f"{item[1]}")
            file.write("\n")


def write_test_fasta_dir(df, test_dir) -> None:
    id_seqs = df["identifier"].values
    sequences = df["sequence"].values

    os.mkdir(test_dir)

    for i in range(len(id_seqs)):
        name_seq = id_seqs[i] + ".fasta"
        seq = sequences[i]

        with open(f"{test_dir}/{name_seq}", "w") as fi:
            fi.write(">" + id_seqs[i])
            fi.write("\n")

            fi.write(seq)
            fi.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert a CSV file to a FASTA file")
    parser.add_argument("train_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("test_csv", type=str, help="Path to the input CSV file")

    args = parser.parse_args()

    # Read the CSV file
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_fasta = os.path.basename(args.train_csv).replace(".csv", ".fasta")
    test_dir = os.path.basename(args.test_csv).replace(".csv", "")

    write_train_fasta(train_df, train_fasta)
    write_test_fasta_dir(test_df, test_dir)


if __name__ == "__main__":
    main()
