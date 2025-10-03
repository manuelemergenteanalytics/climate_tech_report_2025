"""Prospect index builder based on interest scores."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

INTEREST_PATH = Path("data/processed/interest_scores.csv")
PROSPECT_PATH = Path("data/processed/prospect_index.csv")
SOLAR_OUTPUT = Path("data/processed/prospects_solar_co.csv")
DIGITAL_OUTPUT = Path("data/processed/prospects_digital_assets_ar.csv")


def _load_interest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Corré `ctr25 compute-iic` para generar interest_scores.csv."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("interest_scores.csv está vacío")
    return df


def _prospect_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "company_id",
        "company_name",
        "country",
        "industry",
        "size_bin",
        "score_0_100",
        "signals_count",
        "last_ts",
    ]
    group_cols = [c for c in df.columns if c.startswith("group_")]
    cols.extend(group_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"interest_scores.csv carece de columnas: {missing}")
    return df[cols].copy()


def _write_targeted(df: pd.DataFrame, *, output: Path, countries: Tuple[str, ...], industries: Tuple[str, ...], limit: int = 25) -> None:
    subset = df[df["country"].isin(countries) & df["industry"].isin(industries)].copy()
    subset = subset[subset["score_0_100"] >= 40]
    subset = subset.sort_values("score_0_100", ascending=False).head(limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False)


def compute_ps(
    interest_path: str | Path = INTEREST_PATH,
    out_path: str | Path = PROSPECT_PATH,
) -> str:
    interest = _load_interest(Path(interest_path))
    df = _prospect_columns(interest)
    df = df.sort_values("score_0_100", ascending=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Targeted views
    _write_targeted(
        df,
        output=SOLAR_OUTPUT,
        countries=("CO",),
        industries=("energy_power", "water_waste_circularity"),
    )
    _write_targeted(
        df,
        output=DIGITAL_OUTPUT,
        countries=("AR",),
        industries=("finance_insurance", "ict_telecom"),
    )
    return str(out_path)


def compute_ps_cli() -> None:
    path = compute_ps()
    print(f"[compute-ps] wrote {path}")
