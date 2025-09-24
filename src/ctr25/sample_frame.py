import math
from pathlib import Path
import pandas as pd
import yaml
import numpy as np


def _load_project():
    with open("config/project.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_universe() -> pd.DataFrame:
    """Carga el universo. Usa processed si existe; si no, cae al mock."""
    p_processed = Path("data/processed/universe.csv")
    p_mock = Path("data/samples/universe_mock.csv")
    if p_processed.exists():
        return pd.read_csv(p_processed)
    if p_mock.exists():
        return pd.read_csv(p_mock)
    raise FileNotFoundError("No se encontró universo: data/processed/universe.csv ni data/samples/universe_mock.csv")


def _proportional_allocation(Nh: pd.Series, total_target: int, min_per_stratum: int) -> pd.Series:
    """Devuelve n_h enteros: piso por estrato + reparto proporcional del remanente."""
    H = len(Nh)
    # piso inicial
    n_min = pd.Series(min_per_stratum, index=Nh.index, dtype=float)
    # cap por tamaño de universo (no podés muestrear más que N_h)
    n_min = np.minimum(n_min, Nh).astype(float)

    remaining = total_target - int(n_min.sum())
    if remaining <= 0:
        # ya cumplimos, redondeo final
        return n_min.astype(int)

    # proporciones sobre capacidad disponible
    capacity = (Nh - n_min).clip(lower=0.0)
    if capacity.sum() == 0:
        return n_min.astype(int)

    prop = capacity / capacity.sum()
    extra = (prop * remaining).round()

    # ajuste por diferencias de redondeo
    diff = remaining - int(extra.sum())
    if diff != 0:
        # asigna unidades faltantes/sobrantes a los estratos con mayor residuo fraccional
        residual = (prop * remaining) - extra
        order = residual.sort_values(ascending=(diff < 0)).index.tolist()
        for i in range(abs(diff)):
            extra[order[i % len(order)]] += 1 if diff > 0 else -1

    nh = n_min + extra
    # cap final por N_h
    nh = np.minimum(nh, Nh).astype(int)
    return nh


def build_sample():
    """
    Lee config.sample, estratifica por country×industry,
    y escribe data/processed/universe_sample.csv con weight_stratum.
    """
    cfg = _load_project()
    s = (cfg.get("sample") or {})
    total_target = int(s.get("total_target", 1200))
    min_per_stratum = int(s.get("min_per_stratum", 20))
    stratify_by = list(s.get("stratify_by", ["country", "industry"]))
    seed = int(s.get("seed", 42))

    uni = _load_universe().copy()

    # chequeos mínimos
    missing_cols = [c for c in (stratify_by + ["company_id", "company_name"]) if c not in uni.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en universo: {missing_cols}")

    # tabla de tamaños por estrato
    g = uni.groupby(stratify_by, dropna=False)
    Nh = g.size().rename("N_h")

    # asignación n_h
    nh = _proportional_allocation(Nh, total_target, min_per_stratum).rename("n_h")

    # muestreo por estrato (reproducible)
    rng = np.random.default_rng(seed)
    parts = []
    for key, df_h in g:
        k = key if isinstance(key, tuple) else (key,)
        N_h = len(df_h)
        n_h = int(nh.loc[k])
        if n_h <= 0:
            continue
        # sample sin reemplazo (si n_h == N_h, toma todo)
        idx = rng.choice(df_h.index.to_numpy(), size=n_h, replace=False) if n_h < N_h else df_h.index.to_numpy()
        part = df_h.loc[idx].copy()
        part["N_h"] = N_h
        part["n_h"] = n_h
        parts.append(part)

    sample_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(uni.columns) + ["N_h", "n_h"])

    # weight por estrato
    sample_df["weight_stratum"] = sample_df["N_h"] / sample_df["n_h"]

    out = Path("data/processed/universe_sample.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out, index=False)
    print(f"Muestra estratificada → {out} (n={len(sample_df)})")
