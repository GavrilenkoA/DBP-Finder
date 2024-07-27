import pandas as pd
import glob
from tqdm import tqdm
import argparse


def process_phmmer_outputs(train_csv: str, test_csv: str, phmmer_dir: str) -> None:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    phmmer_outputs = glob.glob(f"{phmmer_dir}/*.txt")

    results = {"identifier": [], "metric": [], "found_hit": [], "score": []}

    for file in tqdm(phmmer_outputs, total=len(phmmer_outputs)):
        hits = []
        idx_start = float("inf")

        with open(file, "r") as fi:
            for i, line in enumerate(fi):
                if "Query:" in line:
                    test_seq = line.split()[1]
                elif "Domain annotation for each sequence (and alignments)" in line:
                    break
                elif "E-value  score  bias    E-value  score  bias    exp  N  Sequence" in line:
                    idx_start = i + 2
                elif i >= idx_start:
                    hits.append(line)

        hits = [item.strip() for item in hits if "------ inclusion threshold" not in item and item.strip()]

        if "No hits detected that satisfy reporting thresholds" in hits[0]:
            results["identifier"].append(test_seq)
            results["metric"].append(0)
            results["found_hit"].append("NO")
            results["score"].append(0)

        else:
            results["identifier"].append(test_seq)
            results["found_hit"].append("YES")

            items = hits[0].split()

            score = float(items[1])
            results["score"].append(score)

            nearby_train_seq = items[-1]
            nearby_family = train_df.query(f"identifier == '{nearby_train_seq}'")["label"].iloc[0]
            ground_truth_family = test_df.query(f"identifier == '{test_seq}'")["label"].iloc[0]

            if nearby_family == ground_truth_family:
                results["metric"].append(1)
            else:
                results["metric"].append(0)

    results = pd.DataFrame(results)

    phmmer_metrics_csv = phmmer_dir + ".csv"
    results.to_csv(phmmer_metrics_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Process PHMMER outputs")
    parser.add_argument("train_csv", type=str, help="Path to the training CSV file")
    parser.add_argument("test_csv", type=str, help="Path to the testing CSV file")
    parser.add_argument("phmmer_dir", type=str, help="Directory containing PHMMER output files")

    args = parser.parse_args()

    process_phmmer_outputs(args.train_csv, args.test_csv, args.phmmer_dir)


if __name__ == "__main__":
    main()
