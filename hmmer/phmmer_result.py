import pandas as pd
import glob
import logging

train_csv = input()
test_csv = input()
phmmer_dir = input()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"{phmmer_dir}.log",
)


def process_phmmer_outputs(train_csv: str, test_csv: str, phmmer_dir: str) -> None:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    phmmer_outputs = glob.glob(f"{phmmer_dir}/*.txt")

    results = {"identifier": [], "metric": [], "found_hit": []}

    for j, file in enumerate(phmmer_outputs):
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

        else:
            results["identifier"].append(test_seq)
            results["found_hit"].append("YES")

            nearby_train_seq = hits[0].split()[-1]

            nearby_family = train_df.query(f"identifier == '{nearby_train_seq}'")["label"].iloc[0]
            ground_truth_family = test_df.query(f"identifier == '{test_seq}'")["label"].iloc[0]

            if nearby_family == ground_truth_family:
                results["metric"].append(1)
            else:
                results["metric"].append(0)

        logging.info(f"{test_seq} - processed")

    results = pd.DataFrame(results)

    phmmer_metrics_csv = phmmer_dir + ".csv"
    results.to_csv(phmmer_metrics_csv, index=False)


process_phmmer_outputs(train_csv, test_csv, phmmer_dir)