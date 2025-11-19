#!/usr/bin/env python3
"""Build targeted prospect lists using weighted signal rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import argparse

import pandas as pd
import yaml

from market_radar_theme import apply_industry_labels


DATA_DIR = Path("data/processed")
DEFAULT_EVENTS = DATA_DIR / "events_normalized.reclass.csv"
FALLBACK_EVENTS = DATA_DIR / "events_normalized.csv"
CONFIG_PATH = Path("config/weights_prospects.yml")

OUTPUTS = {
    "nativas": DATA_DIR / "prospects_nativas.csv",
    "erco": DATA_DIR / "prospects_erco.csv",
}

RECENCY_CUTOFF = pd.Timestamp(2020, 10, 1, tz="UTC")


def _resolve_events_path(path: Path | None = None) -> Path:
    if path:
        return path
    if DEFAULT_EVENTS.exists():
        return DEFAULT_EVENTS
    return FALLBACK_EVENTS


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la configuración de prospectos: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("weights_prospects.yml debe contener un mapeo YAML")
    return data


def _build_weight_map(tiers: dict) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for tier in tiers.values():
        weight = float(tier.get("weight", 0.0))
        for key in tier.get("countries", []) or tier.get("industries", []):
            if isinstance(key, str):
                mapping[key.strip().upper()] = weight
    return mapping


def _normalize_signal(name: str) -> str:
    name = (name or "").strip().lower()
    if name in {"bcorp", "b corp", "b-corp"}:
        return "b_corp"
    return name


@dataclass
class CompanySignals:
    company_name: str
    country: str
    industry: str
    event_count: int
    has_sbti: bool
    signal_mix: str
    signal_base_score: float
    signal_boost: float
    volume_factor: float
    last_ts: pd.Timestamp | pd.NaT
    recency_factor: float


def _summarize_company_signals(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    signal_cfg = config.get("signals", {})
    base_weights = signal_cfg.get("base_weights", {})
    boost_factors = signal_cfg.get("boost_factors", {})
    volume_cfg = (config.get("prospecting_rules") or {}).get("volume", {})
    per_event = float(volume_cfg.get("per_event", 0.0))
    max_events = int(volume_cfg.get("max_events", 0))

    events = events.copy()
    events["ts"] = pd.to_datetime(events["ts"], errors="coerce", utc=True)
    events["signal_type_norm"] = events["signal_type"].fillna("").map(_normalize_signal)
    events["industry_friendly"] = apply_industry_labels(events["industry"])

    grouped = []
    for company_name, group in events.groupby("company_name"):
        if not isinstance(company_name, str) or not company_name.strip():
            continue
        signal_counts = group["signal_type_norm"].value_counts()
        event_count = int(signal_counts.sum())
        has_sbti = signal_counts.get("sbti", 0) > 0
        mix_parts = [f"{sig}:{int(cnt)}" for sig, cnt in signal_counts.items()]
        signal_mix = ", ".join(mix_parts) if mix_parts else "sin señales"

        def weighted_avg(weights: dict) -> float:
            total = 0.0
            weight_sum = 0.0
            for sig, cnt in signal_counts.items():
                w = float(weights.get(sig, 0.0))
                if w <= 0:
                    continue
                total += w * cnt
                weight_sum += cnt
            return total / weight_sum if weight_sum else 0.0

        signal_base = weighted_avg(base_weights)
        boost = weighted_avg(boost_factors) or 1.0
        volume_multiplier = 1.0 + min(event_count, max_events) * per_event
        last_ts = pd.to_datetime(group["ts"], errors="coerce", utc=True).max()
        recency_factor = 1.0 if pd.notna(last_ts) and last_ts >= RECENCY_CUTOFF else 0.6

        country = group["country"].dropna().astype(str).str.upper().mode()
        country_value = country.iat[0] if not country.empty else ""
        industry = group["industry_friendly"].dropna().astype(str).mode()
        industry_value = industry.iat[0] if not industry.empty else "Sin industria"

        grouped.append(
            CompanySignals(
                company_name=company_name.strip(),
                country=country_value,
                industry=industry_value,
                event_count=event_count,
                has_sbti=has_sbti,
                signal_mix=signal_mix,
                signal_base_score=signal_base,
                signal_boost=boost,
                volume_factor=volume_multiplier,
                last_ts=last_ts,
                recency_factor=recency_factor,
            )
        )

    return pd.DataFrame([vars(item) for item in grouped])


def _weight_for(value: str, mapping: Dict[str, float]) -> float:
    if not value:
        return 0.0
    return float(mapping.get(value.upper().strip(), 0.0))


def _format_justification(row: pd.Series, country_w: float, industry_w: float) -> str:
    last_ts = row.get("last_ts")
    if isinstance(last_ts, pd.Timestamp) and not pd.isna(last_ts):
        last_ts_fmt = last_ts.date().isoformat()
    else:
        last_ts_fmt = "sin fecha"
    return (
        f"{row['event_count']} señales ({row['signal_mix']}); "
        f"país={row['country']} (w={country_w:.1f}); "
        f"industria={row['industry_friendly']} (w={industry_w:.1f}); "
        f"mix={row['signal_base_score']:.2f} · boost={row['signal_boost']:.2f} · volumen={row['volume_factor']:.2f} · "
        f"recencia={row['recency_factor']:.2f} (última: {last_ts_fmt})"
    )


def _score_target(companies: pd.DataFrame, config: dict, target: str) -> pd.DataFrame:
    weights_cfg = config.get("prospecting_weights", {}).get(target, {})
    if not weights_cfg:
        raise ValueError(f"No hay pesos configurados para el target '{target}'")

    country_map = _build_weight_map(weights_cfg.get("countries", {}))
    industry_map = _build_weight_map(weights_cfg.get("industries", {}))

    rules = config.get("prospecting_rules", {})
    min_country = float(rules.get("min_country_weight", 0.0))
    min_industry = float(rules.get("min_industry_weight", 0.0))

    df = companies.copy()
    df["industry_friendly"] = df["industry"]
    df["country_weight"] = df["country"].map(lambda x: _weight_for(x, country_map))
    df["industry_weight"] = df["industry_friendly"].map(lambda x: _weight_for(x, industry_map))

    mask = (df["country_weight"] >= min_country) & (df["industry_weight"] >= min_industry)
    df = df[mask].copy()
    if df.empty:
        return df

    df["score"] = (
        df["signal_base_score"].clip(lower=0.0)
        * df["signal_boost"].clip(lower=1.0)
        * df["volume_factor"].clip(lower=1.0)
        * df["recency_factor"].clip(lower=0.1)
        * df["country_weight"]
        * df["industry_weight"]
    )
    df["justification"] = df.apply(
        lambda row: _format_justification(row, row["country_weight"], row["industry_weight"]), axis=1
    )
    df = df.sort_values("score", ascending=False).head(50).copy()
    return df[
        [
            "company_name",
            "country",
            "industry_friendly",
            "signal_mix",
            "has_sbti",
            "event_count",
            "score",
            "justification",
        ]
    ]


def build_prospect_lists(events_path: Path, config_path: Path) -> Tuple[str, str]:
    config = _load_config(config_path)
    events = pd.read_csv(events_path)
    if events.empty:
        raise ValueError("events_normalized.csv está vacío; no se pueden generar prospectos")

    companies = _summarize_company_signals(events, config)
    if companies.empty:
        raise ValueError("No se detectaron compañías válidas para prospectar")

    outputs: Dict[str, str] = {}
    for target, output_path in OUTPUTS.items():
        scored = _score_target(companies, config, target)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        outputs[target] = str(output_path)
    return outputs["nativas"], outputs["erco"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera listas de prospectos usando pesos personalizados")
    parser.add_argument("--events", type=Path, default=None, help="CSV de events_normalized[_reclass].csv")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Archivo weights_prospects.yml")
    args = parser.parse_args()

    events_path = _resolve_events_path(args.events)
    nativas_path, erco_path = build_prospect_lists(events_path, args.config)
    print(f"[prospect-score] wrote {nativas_path} and {erco_path}")


if __name__ == "__main__":
    main()
