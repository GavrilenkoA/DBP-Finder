import clearml
from clearml import Logger
import joblib
from clearml import Task as clearml_task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import pandas as pd
from train_src import Metrics, EmbeddingDataset, get_embed_clustered_df, plot_roc_curve


SEED = 42
pd.set_option('display.float_format', '{:.9f}'.format)


def initialize_task(test_data):
    clearml.browser_login()
    task_ml = clearml_task.init(
        project_name="DBPs_search",
        task_name=f"{test_data}_LightAutoML",
        output_uri=True,
    )
    logger = Logger.current_logger()
    return task_ml, logger


def save_model(model, test_data):
    joblib.dump(model, f"models/{test_data}.pkl")


def load_data(test_data):
    train = get_embed_clustered_df(
        embedding_path="../../../ssd2/dbp_finder/ankh_embeddings/train_p3_2d.h5",
        csv_path=f"data/splits/train_{test_data}.csv",
    )

    test = get_embed_clustered_df(
        embedding_path=f"../../../ssd2/dbp_finder/ankh_embeddings/{test_data}_2d.h5",
        csv_path=f"data/embeddings/input_csv/{test_data}.csv",
    )
    return train, test


def prepare_data(train, test):
    train_dataset = EmbeddingDataset(train)
    df_train = train_dataset.make_lama_df()
    y_train = df_train["label"].values

    test_dataset = EmbeddingDataset(test)
    df_test = test_dataset.make_lama_df()
    y_test = df_test["label"].values

    return df_train, y_train, df_test, y_test


def train_model(df_train, y_train, seed=SEED):
    roles = {"target": "label", "group": "cluster"}
    task = Task("binary")

    automl = TabularAutoML(task=task, reader_params={"random_state": seed})
    oof_pred = automl.fit_predict(df_train, roles=roles)

    valid_prob = oof_pred.data[:, 0]
    valid_pred = (valid_prob > 0.5) * 1
    metrics = Metrics(y_train, valid_pred, valid_prob, "valid")
    return automl, metrics.get_metrics()


def test_model(automl, df_test, y_test, test_data):
    test_pred = automl.predict(df_test)
    test_prob = test_pred.data.reshape(-1, )
    test_pred = (test_pred.data[:, 0] > 0.5) * 1

    metrics = Metrics(y_test, test_pred, test_prob, test_data)
    test_metrics = metrics.get_metrics()
    return test_metrics, test_prob


def log_metrics(logger, valid_metrics, test_metrics, y_test, test_prob, test_data):
    logger.report_table(
        title="Validation metrics", series="", table_plot=valid_metrics
    )

    logger.report_table(
        title="Test metrics", series="", table_plot=test_metrics
    )

    plot_roc_curve(y_test, test_prob, test_data)


def main():
    test_data = input("Enter test data name: ")

    # Step 1: Initialize Task and Logger
    task_ml, logger = initialize_task(test_data)

    # Step 2: Load Data
    train, test = load_data(test_data)

    # Step 3: Prepare Data
    df_train, y_train, df_test, y_test = prepare_data(train, test)

    logger.report_table(
        title="LightAutoML Dataframe", series="train", table_plot=df_train.head(4)
    )

    # Step 4: Train Model
    automl, valid_metrics = train_model(df_train, y_train)

    # Step 5: Test Model
    test_metrics, test_prob = test_model(automl, df_test, y_test, test_data)

    # Step 6: Log Metrics and Plot ROC Curve
    log_metrics(logger, valid_metrics, test_metrics, y_test, test_prob, test_data)

    save_model(automl, test_data)


if __name__ == "__main__":
    main()
