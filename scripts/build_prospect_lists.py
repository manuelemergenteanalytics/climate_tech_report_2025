#!/usr/bin/env python3
"""Regenerate targeted prospect lists using enriched events."""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

EVENTS_PATH = Path("data/processed/events_normalized.csv")
RENEWABLES_PATH = Path("data/processed/prospects_renewables.csv")
DIGITAL_PATH = Path("data/processed/prospects_digital_assets.csv")

RENEWABLES_INDUSTRIES = {
    "energy_power_utilities",
    "environmental_circular_services",
}
DIGITAL_INDUSTRIES = {
    "ict_digital_media",
    "finance_insurance_capital",
    "professional_services_consulting",
}

RENEWABLES_KEYWORDS = (
    "renewable",
    "energia",
    "energy",
    "solar",
    "wind",
    "hidrogeno",
    "hydrogen",
    "fotovolta",
    "photovolta",
    "biofuel",
    "clean energy",
    "geothermal",
    "net zero",
)
DIGITAL_KEYWORDS = (
    "digital",
    "data",
    "cloud",
    "plataforma",
    "platform",
    "blockchain",
    "ai ",
    "artificial intelligence",
    "iot",
    "analytics",
    "software",
    "cyber",
    "smart",
)


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    norm = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _keyword_mask(series: pd.Series, keywords: Iterable[str]) -> pd.Series:
    prepared = [_normalize(kw) for kw in keywords if kw]
    if not prepared:
        return pd.Series([False] * len(series), index=series.index)
    normalized = series.fillna("").map(_normalize)
    mask = pd.Series(False, index=series.index)
    for kw in prepared:
        if not kw:
            continue
        mask = mask | normalized.str.contains(kw, na=False)
    return mask


def _filter_candidates(df: pd.DataFrame, *, industries: Iterable[str], keywords: Iterable[str]) -> pd.DataFrame:
    industries = set(industries)
    industry_mask = df["industry"].isin(industries)
    keyword_mask = _keyword_mask(
        df[["title", "description", "text_snippet"]].fillna("").agg(" ".join, axis=1),
        keywords,
    )
    return df[industry_mask | keyword_mask].copy()


def _is_external(company_id: object) -> bool:
    try:
        return not str(int(company_id)).isdigit()
    except (ValueError, TypeError):
        return True


def _build_justification(group: pd.DataFrame) -> str:
    signals = sorted(set(group["signal_type"].fillna("unknown")))
    last_ts = pd.to_datetime(group.get("ts"), errors="coerce").max()
    details: List[str] = []
    if signals:
        details.append(f"signals: {', '.join(signals)}")
    if isinstance(last_ts, pd.Timestamp) and not pd.isna(last_ts):
        details.append(f"last_update: {last_ts.date().isoformat()}")
    snippet = next((s for s in group["description"] if isinstance(s, str) and s.strip()), "")
    if not snippet:
        snippet = next((s for s in group["text_snippet"] if isinstance(s, str) and s.strip()), "")
    snippet = snippet.strip()
    if snippet:
        snippet = re.sub(r"<[^>]+>", " ", snippet)
        snippet = snippet.replace("\n", " ")
        details.append(snippet[:220])
    return " | ".join(details)[:320]


def _summarize(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["company_name", "country", "industry", "source", "justification"])

    grouped = []
    for company_name, group in df.groupby("company_name"):
        group = group.sort_values("ts", ascending=False)
        country = group["country"].dropna().astype(str).str.upper().mode().iat[0]
        industry = group["industry"].mode().iat[0]
        sources = "|".join(sorted(set(group["signal_type"].fillna("unknown"))))
        justification = _build_justification(group)
        last_ts = pd.to_datetime(group["ts"], errors="coerce").max()
        grouped.append(
            {
                "company_name": company_name,
                "country": country,
                "industry": industry,
                "source": sources,
                "justification": justification,
                "_last_ts": last_ts,
                "_signal_count": group["signal_type"].nunique(),
            }
        )

    result = pd.DataFrame(grouped)
    result = result.sort_values(
        ["_signal_count", "_last_ts", "company_name"],
        ascending=[False, False, True],
    ).head(limit)
    return result.drop(columns=["_last_ts", "_signal_count"])


def rebuild_lists(events_path: str | Path = EVENTS_PATH) -> Tuple[str, str]:
    events = pd.read_csv(events_path)
    events["company_id"] = events["company_id"].astype(str)
    events["ts"] = pd.to_datetime(events["ts"], errors="coerce")
    external = events[events["company_id"].apply(_is_external)].copy()
    if external.empty:
        raise ValueError("No external events available to build prospects.")

    renewables = _filter_candidates(
        external,
        industries=RENEWABLES_INDUSTRIES,
        keywords=RENEWABLES_KEYWORDS,
    )
    digital = _filter_candidates(
        external,
        industries=DIGITAL_INDUSTRIES,
        keywords=DIGITAL_KEYWORDS,
    )

    renewables_df = _summarize(renewables)
    digital_df = _summarize(digital)

    RENEWABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    renewables_df.to_csv(RENEWABLES_PATH, index=False)
    digital_df.to_csv(DIGITAL_PATH, index=False)
    return str(RENEWABLES_PATH), str(DIGITAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild prospect lists from events")
    parser.add_argument("--events", default=str(EVENTS_PATH), help="Path to events_normalized.csv")
    args = parser.parse_args()
    renewables_path, digital_path = rebuild_lists(args.events)
    print(f"[prospects] wrote {renewables_path} and {digital_path}")


if __name__ == "__main__":
    main()
