# DBP-Finder

## Introduction
DBP-Finder is an advanced deep learning tool designed for the precise identification of DNA-binding proteins (DBPs). Leveraging the power of state-of-the-art pretrained language models (PLMs), particularly Ankh, DBP-Finder accurately predicts DNA-binding proteins from protein sequence data. This tool empowers researchers to gain deeper insights into genetic regulation and protein function.


##  Installation

### Environment Setup:
```bash
git clone --depth 1 https://github.com/GavrilenkoA/DBP-Finder.git
cd DBP-Finder
conda env create -f env.yaml
conda activate DBP-Finder
```

## Data

### Preparation training dataset
The training dataset is created using the following script, which processes positive and negative samples based on provided metadata:
```bash
python3 -m scripts.data.prepare_data --binders_path <binders.fasta> --non_binders_path <non_binders.fasta> --path_yml <path.yml> --output_path <output.csv>
```
#### Arguments:
1. `--binders_path`:

    * Type: str.
    * Default: "data/uniprot/go_0003677_swissprot.fasta"
    * Description: Specifies the file path to the FASTA file containing the positive samples (i.e., DNA-binding proteins, labeled with GO:0003677).

2. `--non_binders_path`:

    * Type: str.
    * Default: "data/uniprot/notgo_0003723_notgo_0003677_swissprot.fasta"
    * Description: Specifies the file path to the FASTA file containing the negative samples (i.e., proteins that are neither RNA-binding nor DNA-binding).
3. `--path_yml`:

    * Type: str.
    * Default: "data/processed/neg_samples_annot.yml"
    * Description: Specifies the file path to a YML file that includes metadata on whether any negative samples may bind to nucleic acids (GO:0003676). This file helps in further filtering or analyzing the negative samples.

4. `--output_path`:

    * Type: str.
    * Default: "data/embeddings/input_csv/train_p3.csv".
    * Description: Specifies the file path where the final processed dataset, including both positive and negative samples, will be saved as a CSV file.


## Usage

This script performs inference to predict DNA-binding proteins (DBPs) from sequences provided in a FASTA Uniprot file using Ankh pretrained protein language model.

### Command

To execute the script, use the following command:

```bash
python3 -m training-pLM.inference <FASTA_FILE> <OUTPUT_NAME> [--gpu <GPU_ID>]
```

__Example__

From the ROOT directory:

```bash
python3 -m training-pLM.inference data/fasta/Human-dna-binders.fasta Human-dna-binders --gpu 0
```

__Output__

The output will be saved in the `data/prediction` directory with the filename `Human-dna-binders.csv`

__Input Requirements__

Input sequences must be in FASTA __UniProt__ format.
Sequences should only contain canonical amino acids:

`S, N, Y, L, R, Q, D, P, M, F, C, E, W, G, T, K, I, V, A, H`

Sequences must have lengths between 50 and 1024 characters.
Duplicate sequences and identifiers will be automatically removed.

__Logs and Warnings__

The script provides detailed logging to inform you about:

* Sequences with invalid characters that are removed.
* Sequences that do not meet length criteria.
* Duplicate sequences or identifiers that are filtered out.
