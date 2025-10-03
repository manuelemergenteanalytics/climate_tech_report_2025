"""Utilities to load and filter the sampled universe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

UNIVERSE_PATH = Path("data/processed/universe_sample.csv")


@dataclass
class UniverseFilters:
    countries: Optional[Iterable[str]] = None
    industries: Optional[Iterable[str]] = None
    size_bins: Optional[Iterable[str]] = None


def load_universe(
    path: str | None = None,
    *,
    filters: UniverseFilters | None = None,
    max_companies: int = 0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Load the sampled universe applying optional filters and shuffling."""
    universe_path = Path(path or UNIVERSE_PATH)
    if not universe_path.exists():
        raise FileNotFoundError(
            f"No se encontró {universe_path}. Corré `ctr25 sample` antes de recolectar señales."
        )
    df = pd.read_csv(universe_path)
    if df.empty:
        return df

    if filters:
        if filters.countries:
            countries = {c.upper() for c in filters.countries}
            df = df[df["country"].str.upper().isin(countries)]
        if filters.industries:
            allowed = {i for i in filters.industries}
            df = df[df["industry"].isin(allowed)]
        if filters.size_bins:
            allowed = {s.lower() for s in filters.size_bins}
            df = df[df["size_bin"].str.lower().isin(allowed)]

    if df.empty:
        return df

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_companies and max_companies > 0:
        df = df.head(max_companies)
    return df
