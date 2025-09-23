import pandas as pd
from pathlib import Path


def build_mock():
samples = pd.read_csv("data/samples/universe_mock.csv")
# En real, aquí vendría la construcción del marco muestral y la selección aleatoria por estrato.
out = Path("data/processed/universe.csv")
samples.to_csv(out, index=False)
print(f"Universe mock → {out}")