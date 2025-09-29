from __future__ import annotations
import os, math
from datetime import datetime, timedelta, timezone
import pandas as pd

IIC_IN = "data/processed/iic_by_company.csv"
EVENTS = "data/processed/events_normalized.csv"
OUT = "data/processed/prospect_score.csv"

HOT_WINDOW_DAYS = 90
ALPHA = 0.6  # peso de IIC
BETA  = 0.4  # peso de señales calientes

def _norm01_by_stratum(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def _f(g: pd.DataFrame) -> pd.DataFrame:
        x = g[col]
        if x.max() == x.min():
            g[col+"_norm"] = 50.0  # plano → valor medio
        else:
            # percentil rank 0–100 (robusto y monotónico)
            g[col+"_norm"] = (100.0 * (x.rank(method="average") - 1) / (len(x) - 1)).round(2)
        return g
    return df.groupby(["country","industry"], group_keys=False).apply(_f)

def compute_ps(
    iic_path: str = IIC_IN,
    events_path: str = EVENTS,
    out_path: str = OUT,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> str:
    if not os.path.exists(iic_path):
        raise FileNotFoundError(f"Missing IIC file: {iic_path}")
    iic = pd.read_csv(iic_path, dtype=str)
    for c in ["H","C","F","P","X","IIC"]:
        iic[c] = pd.to_numeric(iic[c], errors="coerce").fillna(0.0)

    # señales calientes: suma de strengths en 90 días
    if os.path.exists(events_path):
        e = pd.read_csv(events_path, dtype=str)
        e["ts_dt"] = pd.to_datetime(e["ts"], errors="coerce", utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(days=HOT_WINDOW_DAYS)
        e = e[e["ts_dt"] >= cutoff].copy()
        e["signal_strength"] = pd.to_numeric(e["signal_strength"], errors="coerce").fillna(0.0)
        hot = e.groupby("company_id")["signal_strength"].sum().rename("hot_sum")
    else:
        hot = pd.Series(dtype=float, name="hot_sum")

    df = iic.merge(hot, left_on="company_id", right_index=True, how="left")
    df["hot_sum"] = df["hot_sum"].fillna(0.0)

    # normalizar hot_sum por estrato a 0–100 (percentil rank)
    df = _norm01_by_stratum(df, "hot_sum")
    df["hot_norm"] = df["hot_sum_norm"]; df.drop(columns=["hot_sum_norm"], inplace=True)

    # PS = α·IIC + β·hot_norm
    df["PS"] = (alpha*df["IIC"] + beta*df["hot_norm"]).round(2)

    # etiquetas
    def _bucket(v: float) -> str:
        if v >= 75: return "Muy Alto"
        if v >= 50: return "Alto"
        return "Medio"
    df["PS_label"] = df["PS"].map(_bucket)

    out_cols = ["company_id","company_name","country","industry","size_bin","IIC","hot_norm","PS","PS_label","H","C","F","P","X"]
    df[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    return out_path

# CLI-friendly
def compute_ps_cli():
    path = compute_ps()
    print(f"[compute-ps] wrote {path}")
