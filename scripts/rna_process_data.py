import pandas as pd
from utils import RNADataset, convert_fasta_to_df, filter_df, make_balanced_df


def prepare_training_data(
    binders_path: str = "data/rna/raw/uniprotkb_AND_go_0003723_AND_reviewed_t_2024_06_23.fasta",
    non_binders_path1: str = "data/rna/processed/4_annot_score_neg_cl.csv",
    non_binders_path2: str = "data/rna/processed/5_annot_score_neg_cl.csv",
    output_path: str = "data/rna/processed/train.csv",
) -> None:
    binders: pd.DataFrame = convert_fasta_to_df(binders_path)
    binders["label"] = 1

    non_binders_1: pd.DataFrame = pd.read_csv(non_binders_path1)
    non_binders_2: pd.DataFrame = pd.read_csv(non_binders_path2)
    non_binders: pd.DataFrame = pd.concat([non_binders_1, non_binders_2])
    non_binders["label"] = 0

    train: pd.DataFrame = pd.concat([binders, non_binders])
    train = filter_df(train)
    train = make_balanced_df(train)
    train.to_csv(output_path, index=False)


def load_test_rna_datasets(
    paths: list[str] = [
        "data/rna/raw/561_accending_trP351_trN2819_VaP38_VaN313_TeP52_TeN378_pep_label.csv",
        "data/rna/raw/590_accending_trP206_trN1107_VaP22_VaN123_TeP31_TeN142_pep_label.csv",
        "data/rna/raw/3701_accending_trP437_trN5574_VaP43_VaN695_TeP87_TeN1071_pep_label.csv",
        "data/rna/raw/9606_accending_trP1170_trN8485_VaP126_VaN942_TeP178_TeN1202_pep_label.csv",
    ],
) -> pd.DataFrame:
    datasets = [RNADataset(path) for path in paths]
    dataframes = [dataset.get_data() for dataset in datasets]
    concatenated_df = pd.concat(dataframes)
    return concatenated_df


def load_rna_datasets(
    paths: list[str] = [
        "data/rna/raw/pretrain_accending_trP2392_trN38582_VaP292_VaN4881_TeP298_TeN4889_pep_label.csv",
        "data/rna/raw/561_accending_trP351_trN2819_VaP38_VaN313_TeP52_TeN378_pep_label.csv",
        "data/rna/raw/590_accending_trP206_trN1107_VaP22_VaN123_TeP31_TeN142_pep_label.csv",
        "data/rna/raw/3701_accending_trP437_trN5574_VaP43_VaN695_TeP87_TeN1071_pep_label.csv",
        "data/rna/raw/9606_accending_trP1170_trN8485_VaP126_VaN942_TeP178_TeN1202_pep_label.csv",
    ],
) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in paths]
    overall_df = pd.concat(dfs)

    overall_df.drop("Unnamed: 0", axis=1, inplace=True, errors="ignore")
    overall_df.rename(columns={"Meta": "identifier", "pep": "sequence"}, inplace=True)

    overall_df = overall_df.drop_duplicates(subset=["identifier"])
    overall_df = overall_df.drop_duplicates(subset=["sequence"])

    return overall_df


def make_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_df = load_rna_datasets()
    test_df = load_test_rna_datasets()

    overall_df = overall_df[~overall_df["identifier"].isin(test_df["identifier"])]
    train_df = overall_df[~overall_df["sequence"].isin(test_df["sequence"])]

    return train_df, test_df
