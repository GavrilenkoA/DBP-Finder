import pandas as pd


def get_kingdom(taxon: str | None) -> str:
    template = {
        "Eukaryota Metazoa": "Metazoa",
        "Eukaryota Fungi": "Fungi",
        "Eukaryota Viridiplantae": "Viridiplantae",
    }

    if isinstance(taxon, float):
        return "not_found"
    elif taxon in template:
        return template[taxon]
    elif taxon.startswith("Eukaryota"):
        return "Protists"
    else:
        return taxon.split(" ")[0]


def main():
    input_data = input()
    df = pd.read_csv(f"data/processed/{input_data}_taxon.csv")

    df["kingdom"] = df["taxon_id"].apply(get_kingdom)
    df.drop(columns=["taxon_id"], inplace=True)
    df.to_csv(f"data/processed/{input_data}_kingdom.csv", index=False)


if __name__ == "__main__":
    main()
