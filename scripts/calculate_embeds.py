import pandas as pd
from embeds import get_embeds

input_data = input()

data = pd.read_csv(f"data/not_annotated/{input_data}_v1.csv")

get_embeds(data, f"{input_data}_2d", model_name="ankh")
