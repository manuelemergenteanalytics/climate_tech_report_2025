"""Finance signals derived from public Yahoo Finance endpoints."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

from ctr25.utils.events import append_events, resolve_since
from ctr25.utils.universe import UniverseFilters, load_universe

EVENT_COLUMNS = [
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


def _strength_from_change(change: float) -> float:
    change_abs = abs(change)
    if change_abs >= 0.2:
        return 1.0
    if change_abs >= 0.1:
        return 0.75
    if change_abs >= 0.05:
        return 0.6
    return 0.4


def _period_from_months(months: int) -> str:
    months = max(1, months)
    if months >= 24:
        return "2y"
    if months >= 12:
        return "1y"
    if months >= 6:
        return "6mo"
    if months >= 3:
        return "3mo"
    return "1mo"


def collect_finance(
    *,
    universe_path: str = "data/processed/universe_sample.csv",
    months: int = 12,
    since: str | None = None,
    max_companies: int = 0,
    country: str | None = None,
    industry: str | None = None,
) -> int:
    since_dt = resolve_since(months, since)

    filters = UniverseFilters(
        countries=[country] if country else None,
        industries=[industry] if industry else None,
    )
    universe = load_universe(
        path=universe_path,
        filters=filters,
        max_companies=max_companies,
    )
    if universe.empty or "ticker" not in universe.columns:
        return 0

    universe = universe.dropna(subset=["ticker"])
    universe["ticker"] = universe["ticker"].str.strip()
    universe = universe[universe["ticker"].astype(bool)]
    if universe.empty:
        return 0

    ticker_groups = universe.groupby("ticker")
    period = _period_from_months(months)
    events: List[pd.DataFrame] = []

    for ticker, rows in ticker_groups:
        try:
            hist = yf.download(ticker, period=period, progress=False)
        except Exception:
            continue
        if hist.empty or "Close" not in hist.columns:
            continue
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            continue
        hist.index = pd.to_datetime(hist.index, utc=True)
        hist = hist[hist.index >= since_dt]
        if hist.empty:
            continue
        first_close = float(hist["Close"].iloc[0])
        last_close = float(hist["Close"].iloc[-1])
        if first_close == 0:
            continue
        change = (last_close - first_close) / first_close
        ts_iso = hist.index[-1].isoformat()
        snippet = f"Δp {change*100:.1f}% (period {period})"
        strength = _strength_from_change(change)
        url = f"https://finance.yahoo.com/quote/{ticker}"

        df_rows = []
        for _, company in rows.iterrows():
            df_rows.append({
                "company_id": company.get("company_id", ""),
                "company_name": company.get("company_name", ""),
                "country": company.get("country", ""),
                "industry": company.get("industry", ""),
                "size_bin": company.get("size_bin", ""),
                "source": "finance",
                "signal_type": "finance",
                "signal_strength": strength,
                "ts": ts_iso,
                "url": url,
                "title": f"{ticker} activity",
                "text_snippet": snippet,
            })
        events.append(pd.DataFrame(df_rows, columns=EVENT_COLUMNS))

    if not events:
        return 0

    combined = pd.concat(events, ignore_index=True)
    return append_events(combined, source="finance", signal_type="finance")


def run_collect_finance(**kwargs) -> int:
    return collect_finance(**kwargs)
