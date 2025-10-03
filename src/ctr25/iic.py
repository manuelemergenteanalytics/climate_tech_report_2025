"""Interest score computation (IIC) for CTR25."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from ctr25.utils.events import resolve_since

EVENTS_PATH = Path("data/processed/events_normalized.csv")
UNIVERSE_PATH = Path("data/processed/universe_sample.csv")
INTEREST_PATH = Path("data/processed/interest_scores.csv")
SUMMARY_PATH = Path("data/interim/qa/summary_by_stratum.csv")
TOP10_PATH = Path("data/interim/qa/top10_by_stratum.csv")


@dataclass
class SignalWeight:
    group: str
    weight: float


@dataclass
class ScoringConfig:
    decay_half_life_days: float
    group_caps: Dict[str, float]
    type_weights: Dict[str, SignalWeight]


def _load_config(path: str | Path = "config/signal_map.yml") -> ScoringConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Falta {path}. Configurá signal_map.yml")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    decay_half = float((data.get("decay") or {}).get("half_life_days", 365))
    group_caps: Dict[str, float] = {}
    type_weights: Dict[str, SignalWeight] = {}

    groups = data.get("groups", {}) or {}
    for group_name, cfg in groups.items():
        cap = float(cfg.get("cap", 100))
        group_caps[group_name] = cap
        types = cfg.get("types", {}) or {}
        for signal_type, weight in types.items():
            key = str(signal_type).strip().lower()
            type_weights[key] = SignalWeight(group=group_name, weight=float(weight))

    if not type_weights:
        raise ValueError("signal_map.yml no define pesos por tipo de señal")

    return ScoringConfig(
        decay_half_life_days=decay_half,
        group_caps=group_caps,
        type_weights=type_weights,
    )


def _load_events(path: Path, window_start: pd.Timestamp, cfg: ScoringConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Faltan eventos normalizados: {path}")
    events = pd.read_csv(path)
    if events.empty:
        return events

    events["ts_dt"] = pd.to_datetime(events["ts"], errors="coerce", utc=True)
    events = events.dropna(subset=["ts_dt", "signal_type", "signal_strength", "company_id"])
    events = events[events["ts_dt"] >= window_start]
    if events.empty:
        return events

    events["signal_strength"] = pd.to_numeric(events["signal_strength"], errors="coerce").fillna(0.0)
    events = events[events["signal_strength"] > 0]
    if events.empty:
        return events

    events = events.assign(
        signal_type=lambda df: df["signal_type"].astype(str),
    )
    events["signal_type"] = events["signal_type"].str.strip().str.lower()

    mapped = events["signal_type"].map(
        lambda t: cfg.type_weights.get(t)
    )
    mask = mapped.notnull()
    if not mask.any():
        return pd.DataFrame(columns=events.columns)
    events = events[mask].copy()
    events["group"] = mapped[mask].map(lambda sw: sw.group)
    events["weight"] = mapped[mask].map(lambda sw: sw.weight)
    return events


def _apply_decay(events: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    if events.empty:
        return events
    now = pd.Timestamp.utcnow()
    delta_days = (now - events["ts_dt"]).dt.total_seconds() / 86400.0
    half_life = max(half_life_days, 1e-6)
    decay = np.exp(-delta_days / half_life)
    events["decay"] = decay
    events["event_score"] = events["weight"] * events["signal_strength"] * events["decay"]
    return events


def _aggregate_scores(events: pd.DataFrame, cfg: ScoringConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return (
            pd.DataFrame(columns=["company_id", "group", "score"]),
            pd.DataFrame(columns=["company_id", "signals_count", "last_ts"]),
        )

    group_scores = (
        events.groupby(["company_id", "group"], as_index=False)["event_score"].sum()
    )
    capped_scores = []
    for _, row in group_scores.iterrows():
        cap = cfg.group_caps.get(row["group"], 100.0)
        capped_scores.append(min(row["event_score"], cap))
    group_scores["score"] = capped_scores

    meta = events.groupby("company_id").agg(
        signals_count=("company_id", "size"),
        last_ts=("ts_dt", "max"),
    )
    meta["last_ts"] = meta["last_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = meta.reset_index()
    return group_scores, meta


def _pivot_groups(group_scores: pd.DataFrame) -> pd.DataFrame:
    if group_scores.empty:
        return pd.DataFrame()
    pivot = group_scores.pivot_table(
        index="company_id",
        columns="group",
        values="score",
        fill_value=0.0,
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    pivot.columns = [
        "company_id" if col == "company_id" else f"group_{col}"
        for col in pivot.columns
    ]
    return pivot


def _combine_with_universe(
    universe: pd.DataFrame,
    pivot_scores: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    base = universe[
        ["company_id", "company_name", "country", "industry", "size_bin"]
    ].drop_duplicates()
    df = base.merge(pivot_scores, on="company_id", how="left")
    df = df.merge(meta, on="company_id", how="left")
    return df


def _total_score(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in df.columns if c.startswith("group_")]
    if not score_cols:
        df["score_0_100"] = 0.0
        return df
    df[score_cols] = df[score_cols].fillna(0.0)
    df["score_0_100"] = df[score_cols].sum(axis=1).clip(upper=100.0).round(2)
    return df


def _write_summary(df: pd.DataFrame) -> None:
    if df.empty:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SUMMARY_PATH, index=False)
        df.to_csv(TOP10_PATH, index=False)
        return
    summary = (
        df.groupby(["country", "industry"], as_index=False)
        .agg(
            companies=("company_id", "size"),
            avg_score=("score_0_100", "mean"),
            p90_score=("score_0_100", lambda x: np.percentile(x, 90)),
            high_share=("score_0_100", lambda x: np.mean(x >= 60)),
        )
    )
    summary["avg_score"] = summary["avg_score"].round(2)
    summary["p90_score"] = summary["p90_score"].round(2)
    summary["high_share"] = summary["high_share"].round(3)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    top10 = (
        df.sort_values("score_0_100", ascending=False)
        .groupby(["country", "industry"], as_index=False)
        .head(10)
    )
    top10.to_csv(TOP10_PATH, index=False)


def compute_iic(
    *,
    events_path: str | Path = EVENTS_PATH,
    universe_path: str | Path = UNIVERSE_PATH,
    config_path: str | Path = "config/signal_map.yml",
    months: int = 12,
    since: str | None = None,
) -> str:
    cfg = _load_config(config_path)
    universe_path = Path(universe_path)
    if not universe_path.exists():
        raise FileNotFoundError(f"Falta el universo: {universe_path}")
    universe = pd.read_csv(universe_path)
    if universe.empty:
        raise ValueError("El universo está vacío; corré `ctr25 sample` antes de calcular el IIC.")

    since_dt = resolve_since(months, since)
    since_ts = pd.Timestamp(since_dt)
    if since_ts.tzinfo is None:
        since_ts = since_ts.tz_localize("UTC")
    else:
        since_ts = since_ts.tz_convert("UTC")

    events = _load_events(Path(events_path), since_ts, cfg)
    if events.empty:
        combined = universe[["company_id", "company_name", "country", "industry", "size_bin"]].copy()
        combined["score_0_100"] = 0.0
        combined["signals_count"] = 0
        combined["last_ts"] = ""
        INTEREST_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(INTEREST_PATH, index=False)
        _write_summary(combined)
        return str(INTEREST_PATH)

    events = _apply_decay(events, cfg.decay_half_life_days)
    group_scores, meta = _aggregate_scores(events, cfg)
    pivot_scores = _pivot_groups(group_scores)
    combined = _combine_with_universe(universe, pivot_scores, meta)
    combined = _total_score(combined)
    combined["signals_count"] = combined["signals_count"].fillna(0).astype(int)
    combined["last_ts"] = combined["last_ts"].fillna("")
    combined = combined.sort_values("score_0_100", ascending=False).reset_index(drop=True)

    INTEREST_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(INTEREST_PATH, index=False)
    _write_summary(combined)
    return str(INTEREST_PATH)


def compute_iic_cli() -> None:
    path = compute_iic()
    print(f"[compute-iic] wrote {path}")
