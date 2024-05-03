import streamlit as st
from src.train_src import merge_embed
from src.utils import collect_df, save_csv
from src.train_src import make_inference_lama_df, predict
from src.embeds import get_embeds


def main():
    st.title("DBP-finder demo")

    uploaded_file = st.file_uploader("Upload a fasta file", type=["fasta"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        content = uploaded_file.read().decode()
        df = collect_df(content)
        basename = uploaded_file.name.replace(".fasta", "")

        save_csv(df, basename, path="data/embeddings/input_csv/")
        get_embeds(data_name=basename)
        test_embed = merge_embed(df, f"data/embeddings/ankh_embeddings/{basename}.pkl")
        df_test = make_inference_lama_df(test_embed)
        test_prob, test_pred = predict(df_test)

        df.loc[:, "score"] = test_prob
        df.loc[:, "pred_label"] = test_pred
        st.dataframe(df)


if __name__ == "__main__":
    main()
