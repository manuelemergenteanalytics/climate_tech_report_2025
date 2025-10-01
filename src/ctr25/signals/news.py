"""Simple news collectors (GDELT + RSS)."""
from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Sequence

import feedparser
import pandas as pd
import yaml

from ctr25.utils.events import append_multiple
from ctr25.utils.http import get

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


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _gdelt_ts(value: str | None) -> str:
    if not value:
        return _utc_now_iso()
    value = value.strip()
    try:
        parsed = dt.datetime.strptime(value, "%Y%m%d%H%M%S")
        return parsed.replace(tzinfo=dt.timezone.utc).isoformat()
    except ValueError:
        return _utc_now_iso()


def _rss_ts(value: str | None) -> str:
    if not value:
        return _utc_now_iso()
    try:
        parsed = parsedate_to_datetime(value)
        if not parsed.tzinfo:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).isoformat()
    except Exception:
        return _utc_now_iso()


def _make_event(*, url: str, title: str, snippet: str, ts: str) -> dict:
    return {
        "company_id": "",
        "company_name": "",
        "country": "",
        "industry": "",
        "size_bin": "",
        "source": "news",
        "signal_type": "news",
        "signal_strength": 1.0,
        "ts": ts,
        "url": url,
        "title": title[:300],
        "text_snippet": snippet[:500],
    }


def fetch_gdelt(keywords: Sequence[str], days: int = 7, max_records: int = 250) -> pd.DataFrame:
    if not keywords:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    query_terms = [k for k in keywords if k]
    query = " OR ".join(f'"{term}"' for term in query_terms)
    if query:
        query = f"({query})"
    if not query:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": since,
    }
    response = get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
    try:
        payload = response.json()
    except ValueError:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for article in payload.get("articles", []):
        url = article.get("url") or ""
        if not url:
            continue
        title = article.get("title") or ""
        snippet = article.get("snippet") or ""
        ts = _gdelt_ts(article.get("seendate"))
        rows.append(_make_event(url=url, title=title, snippet=snippet, ts=ts))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def fetch_rss(feeds: Iterable[str], limit_per_feed: int = 50) -> pd.DataFrame:
    rows: list[dict] = []
    for feed_url in feeds:
        if not feed_url:
            continue
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:limit_per_feed]:
            url = getattr(entry, "link", "")
            if not url:
                continue
            title = getattr(entry, "title", "")
            snippet = getattr(entry, "summary", "")
            ts_value = getattr(entry, "published", None) or getattr(entry, "updated", None)
            ts = _rss_ts(ts_value)
            rows.append(_make_event(url=url, title=title, snippet=snippet, ts=ts))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _load_keywords(path: str) -> dict:
    config = {"include": [], "exclude": []}
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        config["include"] = data.get("include", [])
        config["exclude"] = data.get("exclude", [])
    return config


def _load_news_cfg(path: str) -> dict:
    default = {
        "providers": {
            "gdelt": {"enabled": True, "days": 7, "max_records": 250},
            "rss": {"enabled": False, "feeds": [], "limit_per_feed": 50},
        }
    }
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return {**default, **loaded}


def run_collect_news(
    universe_path: str = "data/processed/universe_sample.csv",
    keywords_path: str = "config/keywords.yml",
    aggregator_path: str = "config/news.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
) -> int:
    # universe filters are kept for CLI compatibility but not used directly yet
    keywords_cfg = _load_keywords(keywords_path)
    news_cfg = _load_news_cfg(aggregator_path)

    frames: list[pd.DataFrame] = []
    providers = news_cfg.get("providers", {})

    gdelt_cfg = providers.get("gdelt", {})
    if gdelt_cfg.get("enabled", True):
        frames.append(
            fetch_gdelt(
                keywords_cfg.get("include", []),
                days=int(gdelt_cfg.get("days", 7)),
                max_records=int(gdelt_cfg.get("max_records", 250)),
            )
        )

    rss_cfg = providers.get("rss", {})
    if rss_cfg.get("enabled", False):
        frames.append(
            fetch_rss(
                rss_cfg.get("feeds", []),
                limit_per_feed=int(rss_cfg.get("limit_per_feed", 50)),
            )
        )

    frames = [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return 0

    added = append_multiple(frames)
    return added
