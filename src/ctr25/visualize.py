import pandas as pd
import plotly.express as px
from pathlib import Path




def build_all():
df = pd.read_csv("data/processed/company_prospects.csv")


# Heatmap país × industria (IIC mediano)
heat = df.groupby(["country", "industry"])['IIC'].median().reset_index()
fig_h = px.density_heatmap(heat, x="industry", y="country", z="IIC", nbinsx=20, nbinsy=20, histfunc="avg")
fig_h.write_html("reports/figures/heatmap_country_industry.html", include_plotlyjs="cdn")


# Tabla de cuentas (Top-N por PS)
top = df.sort_values("PS", ascending=False).head(200)
top.to_csv("reports/public/top_companies.csv", index=False)


print("Visualizaciones generadas → reports/figures, reports/public")




def export_prospects(path: str):
df = pd.read_csv("data/processed/company_prospects.csv")
cols = [
"company_name", "country", "industry", "size_bin", "IIC", "PS", "label"
]
Path(path).parent.mkdir(parents=True, exist_ok=True)
df[cols].sort_values("PS", ascending=False).to_csv(path, index=False)
print(f"Prospects exportados → {path}")