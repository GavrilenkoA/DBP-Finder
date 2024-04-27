import argparse
import joblib
from train_src import make_inference_lama_df, merge_embed
from utils import convert_fasta_to_df, save_csv
from embedding_calculate import calculate_embeds
import pandas as pd


def predict(model, df, df_test) -> pd.DataFrame:
    test_pred = model.predict(df_test)
    test_prob = test_pred.data.reshape(-1, )
    test_pred = (test_pred.data[:, 0] > 0.5) * 1

    df.loc[:, "score"] = test_prob
    df.loc[:, "y_pred"] = test_pred
    return df


def main():
    basename = args.i.replace(".fasta", "")

    model = joblib.load("models/DBP-finder.pkl")

    df = convert_fasta_to_df(args.i)
    save_csv(df, basename, path="data/embeddings/input_csv/")
    calculate_embeds(data_name=basename, model_name="ankh")

    test_embed = merge_embed(df, f"data/embeddings/ankh_embeddings/{basename}.pkl")
    df_test = make_inference_lama_df(test_embed)

    df = predict(model, df, df_test)
    save_csv(df, basename + "_prediction", path="data/inference_data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBP-finder inference")
    parser.add_argument("--i", type=str, help="Input fasta", required=True)
    args = parser.parse_args()
    main()
