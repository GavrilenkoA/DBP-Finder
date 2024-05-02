import argparse
from train_src import make_inference_lama_df, merge_embed, predict
from utils import convert_fasta_to_df, save_csv
from embeds import get_embeds


def main():
    parser = argparse.ArgumentParser(description="DBP-finder inference")
    parser.add_argument("--input_fasta", type=str, help="Input fasta", required=True)
    args = parser.parse_args()

    basename = args.input_fasta.replace(".fasta", "")
    file = "data/fasta/" + args.input_fasta

    df = convert_fasta_to_df(file)
    save_csv(df, basename, path="data/embeddings/input_csv/")
    get_embeds(data_name=basename)

    test_embed = merge_embed(df, f"data/embeddings/ankh_embeddings/{basename}.pkl")
    df_test = make_inference_lama_df(test_embed)

    test_prob, test_pred = predict(df_test)
    df.loc[:, "score"] = test_prob
    df.loc[:, "pred_label"] = test_pred
    save_csv(df, basename, path="data/predictions/")


if __name__ == "__main__":
    main()
