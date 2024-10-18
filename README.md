# DBP-Finder

## Introduction
Advanced deep learning tool developed for the precise identification of DNA-binding proteins (DBPs). Using state-of-the-art pretrained language model (Ankh), DBP-finder accurately predicts DBPs from protein sequence data, aiding researchers in understanding genetic regulation and protein function.


##  Installation

### Environment Setup:
```bash
git clone https://github.com/GavrilenkoA/DBP-Finder.git
cd DBP-Finder
conda env create -f env.yaml
conda activate DBP-Finder
```

## Data

### Preparation training dataset
The training dataset named as was formed by running the following script, which processes the positive and negative samples given metadata ..., output
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

### 3.1 Test dataset collection



## Clustering
### Training and test datasets clustering
```bash
python3 scripts/prepare_clustered_data.py <path_train> <path_test>
```
#### Arguments:
1. `path_train`:
    * Type: str.
    * Description: Path to the CSV file containing the training dataset with protein sequences.
2. `path_test`:
    * Type: str.
    * Description: Path to the CSV file containing the test dataset with protein sequences.

__Output__: The final clustered and balanced training dataset is saved as a CSV file in the `data/splits` directory, named based on the input training and test files:
    `data/splits/{name_train}_{name_test}.csv`

### Training dataset clustering
```bash
python3 scripts/get_cluster.py <input_path> <output_path>
```
#### Arguments:
1. `path_train`:
    * Type: str.
    * Description: Path to the CSV file containing the training dataset with protein sequences.
2. `output_path`:
    * Type: str.
    * Description: Path where the output CSV file with added cluster will be saved.

__Example output__: `data/splits/train_p3.csv`
The output file contains additional columns representing the cluster each sequence belongs to


## Calculating Embeddings for Protein Sequences

To calculate embeddings for protein sequences, use the following command in your terminal:

```bash
python3 scripts/calculate_embeddings.py <input_csv> --model_name <model_name> --device <device> --output_prefix <output_prefix>
```
#### Arguments:
1. `input_csv`:
    * Type: str
    * Description: Path to the input CSV file containing protein sequences.
2. `--model_name`:
    * Type: str
    * Default: ankh
    * Description: The name of the model used to calculate embeddings.
3. `--device`:
    * Type: str
    * Default: `cuda:1`
    * Description: The device for calculations, which can be a GPU (e.g., cuda:0) or cpu.
4. `--output_prefix`:
    * Type: str
    * Default: `../../../ssd2/dbp_finder/ankh_embeddings`
    * Description: The directory where the generated embeddings will be saved. The output files will be prefixed with this path.

## Training
### Ankh head
The DBP-Finder model is trained on one node equipped with an A100 GPU (80 GB). To train the model, first navigate to the training directory:

`cd training-pLM`

Then, execute the following command:
```bash
python3 train_ankh_head_full_data.py --embedding_path <path_to_embeddings> --csv_path <path_to_training_csv> --best_model_path <best_model_path> --config <path_to_config_yaml>
```
You can customize the training settings by modifying the `DBP-Finder-config.yml` file as required.


### Ankh Lora

To train the model using the Ankh LoRA configuration, execute:

`python3 train_ankh_lora.py --csv_path <csv_path>
--best_model_path <best_model_path> --lora_config lora_config`

Adjust additional settings in the `lora_config.yml` file as needed.
