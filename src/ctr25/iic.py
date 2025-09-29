from __future__ import annotations
import os, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import yaml

EVENTS = "data/processed/events_normalized.csv"
UNIVERSE = "data/processed/universe_sample.csv"
OUT = "data/processed/iic_by_company.csv"
SIGNAL_MAP_PATH = "config/signal_map.yml"

# pesos IIC
WEIGHTS = {"H":0.25, "C":0.20, "F":0.20, "P":0.20, "X":0.15}
WINDOW_DAYS = 365  # 12m

DEFAULT_MAP = {
  "H": ["job_posting"],
  "C": ["sbti","re100","sistemab","ungc"],
  "F": ["green_bond","sll"],
  "P": ["pilot_news"],
  "X": ["newsroom","web_esg"],
}

def _normcdf01(z: float) -> float:
    # convierte z-score a [0,100] via CDF normal
    return max(0.0, min(100.0, 50.0*(1.0+math.erf(z/(2**0.5)))))

def _winsorize_p95(series: pd.Series) -> pd.Series:
    if series.empty: return series
    p95 = series.quantile(0.95)
    return series.clip(upper=p95)

def _load_signal_map(path: str) -> dict[str, list[str]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            return {k: list(map(str, v or [])) for k, v in y.items()}
    return DEFAULT_MAP

def _pillar_sum(df: pd.DataFrame, pillar_types: list[str]) -> pd.Series:
    if not pillar_types: return pd.Series(0.0, index=df.index)
    mask = df["signal_type"].isin(pillar_types)
    return df.loc[mask].groupby("company_id")["signal_strength"].sum()

def compute_iic(
    events_path: str = EVENTS,
    universe_path: str = UNIVERSE,
    signal_map_path: str = SIGNAL_MAP_PATH,
    out_path: str = OUT,
) -> str:
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing events: {events_path}")
    u = pd.read_csv(universe_path, dtype=str)
    e = pd.read_csv(events_path, dtype=str)

    # parseo de fecha + ventana 12m
    e["ts_dt"] = pd.to_datetime(e["ts"], errors="coerce", utc=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    e = e[e["ts_dt"] >= cutoff].copy()
    if e.empty:
        # salida vacía pero consistente
        out = u[["company_id","company_name","country","industry","size_bin"]].copy()
        for k in WEIGHTS: out[k]=0.0
        out["IIC"]=0.0
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    e["signal_strength"] = pd.to_numeric(e["signal_strength"], errors="coerce").fillna(0.0)

    # map de pilares
    sigmap = _load_signal_map(signal_map_path)

    # sumar por pilar a nivel empresa
    pillars = {}
    for k, types in sigmap.items():
        s = _pillar_sum(e, types)
        pillars[k] = s

    # ensamblar frame por empresa con metadata de estrato
    base = u[["company_id","company_name","country","industry","size_bin"]].drop_duplicates().set_index("company_id")
    for k, s in pillars.items():
        base[k] = s
    base = base.fillna(0.0)

    # winsor p95 SOLO en X antes de estandarizar
    if "X" in base.columns:
        base["X"] = _winsorize_p95(base["X"])

    # estandarización intra-estrato y conversión a 0–100 via CDF normal
    def _scale_group(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        for k in WEIGHTS.keys():
            mu = out[k].mean()
            sd = out[k].std(ddof=0)
            z = 0.0 if sd == 0 else (out[k] - mu) / sd
            out[k] = z.map(_normcdf01)
        return out

    base = base.groupby(["country","industry"], group_keys=False).apply(_scale_group)

    # IIC = suma ponderada
    base["IIC"] = sum(base[k]*w for k, w in WEIGHTS.items())
    base["IIC"] = base["IIC"].round(2)

    # salida
    out = base.reset_index()[["company_id","company_name","country","industry","size_bin","H","C","F","P","X","IIC"]]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

# CLI-friendly
def compute_iic_cli():
    path = compute_iic()
    print(f"[compute-iic] wrote {path}")




