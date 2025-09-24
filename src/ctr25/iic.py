import pandas as pd
import numpy as np
from pathlib import Path
import yaml

PILARS = ["H", "C", "F", "P", "X"]


def _z_by_stratum(df: pd.DataFrame, col: str, by: list[str]) -> pd.Series:
    def z(x: pd.Series) -> pd.Series:
        mu = x.mean()
        sd = x.std(ddof=0) or 1.0
        return (x - mu) / sd

    return df.groupby(by)[col].transform(z)


def compute():
    events = pd.read_csv("data/processed/events_normalized.csv", parse_dates=["ts"])
    with open("config/signal_map.yml", "r", encoding="utf-8") as f:
        map_signal_to_pilar = yaml.safe_load(f)
    events["pilar"] = events["signal_type"].map(map_signal_to_pilar).fillna("X")

    # Agregado: signal_strength por company × pilar
    agg = (
        events.pivot_table(
            index=["company_id", "company_name", "country", "industry", "size_bin"],
            columns="pilar",
            values="signal_strength",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )

    # Asegurar columnas para todos los pilares
    for p in PILARS:
        if p not in agg.columns:
            agg[p] = 0.0

    # Winsorización p95 por pilar
    for p in PILARS:
        q95 = agg[p].quantile(0.95)
        agg[p] = agg[p].clip(upper=q95)

    # Z-score intra-estrato
    by = ["country", "industry", "size_bin"]
    for p in PILARS:
        agg[f"{p}_z"] = _z_by_stratum(agg, p, by)

    # Pesos
    with open("config/weights.yml", "r", encoding="utf-8") as f:
        w = yaml.safe_load(f)

    agg["IIC_raw"] = (
        w["pilar_weights"]["H"] * agg["H_z"]
        + w["pilar_weights"]["C"] * agg["C_z"]
        + w["pilar_weights"]["F"] * agg["F_z"]
        + w["pilar_weights"]["P"] * agg["P_z"]
        + w["pilar_weights"]["X"] * agg["X_z"]
    )

    # Rebase 0–100 por estrato
    def rebase(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.full_like(x, 50.0, dtype=float)
        return 100.0 * (x - mn) / (mx - mn)

    agg["IIC"] = agg.groupby(by)["IIC_raw"].transform(rebase)

    out = Path("data/processed/company_scores.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"IIC computed → {out}")



