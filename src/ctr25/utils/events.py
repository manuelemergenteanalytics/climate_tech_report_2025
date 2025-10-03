"""Helpers to append events into the normalized CSV."""
from __future__ import annotations

from pathlib import Path
import datetime as dt
from typing import Iterable, Optional

import pandas as pd

EVENTS_PATH = Path("data/processed/events_normalized.csv")

_EVENT_COLUMNS: list[str] = [
    "company_id",
    "company_name",
    "country",
    "industry",
    "size_bin",
    "source",
    "signal_type",
    "signal_strength",
    "ts",
    "url",
    "title",
    "text_snippet",
]


def _load_events() -> pd.DataFrame:
    if EVENTS_PATH.exists():
        return pd.read_csv(EVENTS_PATH)
    return pd.DataFrame(columns=_EVENT_COLUMNS)


def resolve_since(months: Optional[int] = 12, since: Optional[str] = None) -> dt.datetime:
    """Compute the earliest timestamp to collect signals from."""
    if since:
        try:
            parsed = dt.datetime.fromisoformat(since)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        except ValueError as exc:
            raise ValueError(f"Formato inválido para --since: {since}") from exc
    months = months or 12
    if months <= 0:
        months = 12
    today = dt.datetime.now(dt.timezone.utc)
    return today - dt.timedelta(days=int(months * 30.5))


def _ensure_columns(df: pd.DataFrame, *, source: str, signal_type: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    defaults = {
        "company_id": "",
        "company_name": "",
        "country": "",
        "industry": "",
        "size_bin": "",
        "source": source,
        "signal_type": signal_type,
        "signal_strength": 1.0,
        "ts": "",
        "url": "",
        "title": "",
        "text_snippet": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    return out[_EVENT_COLUMNS]


def append_events(df: pd.DataFrame, *, source: str, signal_type: str) -> int:
    """Append events to the normalized CSV returning number of new rows."""
    df = _ensure_columns(df, source=source, signal_type=signal_type)
    if df.empty:
        return 0

    current = _load_events()
    before = len(current)
    combined = pd.concat([current, df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["company_id", "signal_type", "url", "ts"], keep="last"
    )
    added = len(combined) - before
    combined.to_csv(EVENTS_PATH, index=False)
    return max(0, added)


def append_multiple(dfs: Iterable[pd.DataFrame]) -> int:
    """Append multiple event DataFrames while preserving dedupe."""
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return 0

    current = _load_events()
    before = len(current)
    merged = pd.concat([current, combined[_EVENT_COLUMNS]], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["company_id", "signal_type", "url", "ts"], keep="last"
    )
    added = len(merged) - before
    merged.to_csv(EVENTS_PATH, index=False)
    return max(0, added)
