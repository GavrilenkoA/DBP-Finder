import pandas as pd
import joblib
from train_src import make_inference_lama_df, merge_embed

test_data = input()

test = pd.read_csv(f"../data/embeddings/input_csv/{test_data}.csv")
test_embed = merge_embed(test, f"../data/embeddings/ankh_embeddings/{test_data}.pkl")
df_test = make_inference_lama_df(test_embed)


automl = joblib.load("models/DBP-finder.pkl")

test_pred = automl.predict(df_test)
test_prob = test_pred.data.reshape(-1, )
test_pred = (test_pred.data[:, 0] > 0.5) * 1

test.loc[:, "score"] = test_prob
test.loc[:, "y_pred"] = test_pred


test.to_csv(f"../data/not_annotated/{test_data}_prediction.csv", index=False)