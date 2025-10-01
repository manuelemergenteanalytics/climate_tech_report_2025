"""Finance signals using Yahoo Finance (yfinance)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

from ctr25.utils.events import append_multiple

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


def _iso_from_timestamp(ts) -> str:
    if isinstance(ts, dt.datetime):
        if ts.tzinfo:
            return ts.astimezone(dt.timezone.utc).isoformat()
        return ts.replace(tzinfo=dt.timezone.utc).isoformat()
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fetch_price_for_ticker(ticker: str):
    try:
        history = yf.Ticker(ticker).history(period="5d")
    except Exception:
        return None
    if history.empty:
        return None
    last = history.tail(1)
    close = float(last["Close"].iloc[0])
    ts_index = last.index[-1]
    ts_iso = _iso_from_timestamp(ts_index.to_pydatetime() if hasattr(ts_index, "to_pydatetime") else ts_index)
    return ts_iso, close


def run_collect_finance(
    universe_path: str = "data/processed/universe_sample.csv",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
) -> int:
    uni_path = Path(universe_path)
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe missing: {universe_path}")
    universe = pd.read_csv(uni_path, dtype=str)
    if "ticker" not in universe.columns:
        return 0

    if country:
        universe = universe[universe["country"] == country]
    if industry:
        universe = universe[universe["industry"] == industry]
    if max_companies > 0:
        universe = universe.head(max_companies)

    tickers = universe["ticker"].dropna().str.strip()
    tickers = tickers[tickers != ""]
    if tickers.empty:
        return 0

    events_rows = []
    grouped = universe[universe["ticker"].isin(tickers)].groupby("ticker")
    for ticker, rows in grouped:
        result = _fetch_price_for_ticker(ticker)
        if not result:
            continue
        ts_iso, close = result
        url = f"https://finance.yahoo.com/quote/{ticker}"
        title = f"{ticker} close={close:.2f}"
        for _, row in rows.iterrows():
            events_rows.append({
                "company_id": row.get("company_id", ""),
                "company_name": row.get("company_name", ""),
                "country": row.get("country", ""),
                "industry": row.get("industry", ""),
                "size_bin": row.get("size_bin", ""),
                "source": "finance",
                "signal_type": "price",
                "signal_strength": 1.0,
                "ts": ts_iso,
                "url": url,
                "title": title,
                "text_snippet": "",
            })

    if not events_rows:
        return 0

    events_df = pd.DataFrame(events_rows, columns=EVENT_COLUMNS)
    added = append_multiple([events_df])
    return added
