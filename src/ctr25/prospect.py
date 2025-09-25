import pandas as pd
import yaml
from pathlib import Path


def compute():
    # Inputs base
    scores = pd.read_csv("data/processed/company_scores.csv")
    events = pd.read_csv("data/processed/events_normalized.csv", parse_dates=["ts"])

    # Config
    with open("config/project.yml", "r", encoding="utf-8") as f:
        project = yaml.safe_load(f)
    with open("config/weights.yml", "r", encoding="utf-8") as f:
        w = yaml.safe_load(f)

    # Ventana de señales "calientes"
    hot_days = project.get("hot_days", 90)
    cutoff = events["ts"].max() - pd.Timedelta(days=hot_days)
    hot = events.loc[events["ts"] >= cutoff]

    # Hot score por empresa
    hot_score = hot.groupby("company_id")["signal_strength"].sum().rename("hot_score")

    # Merge con IIC y fill de NaN
    out = scores.merge(hot_score, on="company_id", how="left").fillna({"hot_score": 0.0})

    # Prospect Score (PS)
    out["PS"] = w["alpha"] * out["IIC"] + w["beta"] * out["hot_score"].rank(pct=True) * 100

    # Etiquetas
    def label(ps: float) -> str:
        if ps >= 67:
            return "Muy Alto"
        if ps >= 40:
            return "Alto"
        return "Medio"

    out["label"] = out["PS"].apply(label)

    # Output
    out_path = Path("data/processed/company_prospects.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"PS computed → {out_path}")
