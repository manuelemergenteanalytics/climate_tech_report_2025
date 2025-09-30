from pathlib import Path

import pandas as pd
import plotly.express as px


PROSPECT_PATH = Path("data/processed/prospect_score.csv")


def _load_prospects() -> pd.DataFrame:
    if not PROSPECT_PATH.exists():
        raise FileNotFoundError(f"Missing prospects file: {PROSPECT_PATH}")
    return pd.read_csv(PROSPECT_PATH)


def build_all():
    df = _load_prospects()

    # Asegurar carpetas de salida
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/public").mkdir(parents=True, exist_ok=True)

    # Heatmap país × industria (IIC mediano)
    heat = df.groupby(["country", "industry"])["IIC"].median().reset_index()
    fig_h = px.density_heatmap(
        heat,
        x="industry",
        y="country",
        z="IIC",
        nbinsx=20,
        nbinsy=20,
        histfunc="avg",
    )
    fig_h.write_html("reports/figures/heatmap_country_industry.html", include_plotlyjs="cdn")

    # Tabla de cuentas (Top-N por PS)
    top = df.sort_values("PS", ascending=False).head(200)
    top.to_csv("reports/public/top_companies.csv", index=False)

    print("Visualizaciones generadas → reports/figures, reports/public")


def export_prospects(path: str):
    df = _load_prospects()
    cols = ["company_name", "country", "industry", "size_bin", "IIC", "PS", "PS_label"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values("PS", ascending=False).to_csv(path, index=False)
    print(f"Prospects exportados → {path}")


def main():
    build_all()
