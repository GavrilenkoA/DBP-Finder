import pandas as pd
from embeds import get_embeds


input_data = input()

data = pd.read_csv(f"data/embeddings/input_csv/{input_data}.csv")

get_embeds(data, f"{input_data}_2d", model_name="ankh")
