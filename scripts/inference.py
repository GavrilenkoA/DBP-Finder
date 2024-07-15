import argparse
import os

import clearml
import pandas as pd
from clearml import Logger
from clearml import Task as clearml_task
from train_src import (Metrics, form_Xy, make_lama_df, merge_embed,
                       plot_roc_curve, predict)


def main():
    parser = argparse.ArgumentParser(description="DBP-finder inference")
    parser.add_argument("--test_data", type=str, help="Input csv", required=True)
    parser.add_argument("--kingdom", type=str, help="Filter by kingdom", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.test_data)
    basename = os.path.basename(args.test_data.replace(".csv", ""))

    test = merge_embed(df, basename)
    X_test, y_test = form_Xy(test)
    df_test = make_lama_df(X_test, y_test)

    test_prob, test_pred = predict(df_test)

    metrics = Metrics(y_test, test_pred, test_prob, basename)
    test_metrics = metrics.get_metrics()

    clearml.browser_login()
    clearml_task.init(
        project_name="DBPs_search",
        task_name=f"{basename}_{args.kingdom}",
        output_uri=True,
    )

    logger = Logger.current_logger()

    logger.report_table(
        title="Test metrics", series="pandas DataFrame", table_plot=test_metrics
    )

    plot_roc_curve(y_test, test_prob, basename)


if __name__ == "__main__":
    main()
