"""News adapters for GDELT + RSS feeds matched to the CTR25 universe."""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import feedparser
import pandas as pd
import requests
import requests_cache
import yaml

from ctr25.utils.events import append_events, resolve_since
from ctr25.utils.keywords import expand_keywords
from ctr25.utils.sentiment import analyze_sentiment
from ctr25.utils.universe import UniverseFilters, load_universe

_CACHE_PATH = Path("data/interim/cache/news_requests.sqlite")
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
requests_cache.install_cache(str(_CACHE_PATH), backend="sqlite", expire_after=3600)

EVENT_COLUMNS = [
    "company_id",
    "company_qid",
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
    "climate_score",
    "sentiment_label",
    "sentiment_score",
]

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class NewsProviders:
    gdelt_enabled: bool = True
    gdelt_days: int = 7
    gdelt_max_records: int = 75
    rss_enabled: bool = False
    rss_feeds: Sequence[str] = ()
    rss_limit_per_feed: int = 50
    google_enabled: bool = False
    google_language: str = "en"
    google_region: str = "US"
    google_limit_per_company: int = 5


@dataclass
class KeywordConfig:
    include: Sequence[str]
    exclude: Sequence[str]


def _load_keywords(path: str) -> KeywordConfig:
    p = Path(path)
    if not p.exists():
        return KeywordConfig(include=(), exclude=())
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    include = tuple(map(str, data.get("include", [])))
    exclude = tuple(map(str, data.get("exclude", [])))
    if include:
        include = tuple(expand_keywords(include))
    return KeywordConfig(include=include, exclude=exclude)


def _load_providers(path: str) -> NewsProviders:
    p = Path(path)
    if not p.exists():
        return NewsProviders()
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    providers = data.get("providers", {})
    gdelt = providers.get("gdelt", {})
    rss = providers.get("rss", {})
    google = providers.get("google_news", {})
    return NewsProviders(
        gdelt_enabled=gdelt.get("enabled", True),
        gdelt_days=int(gdelt.get("days", 7)),
        gdelt_max_records=int(gdelt.get("max_records", 75)),
        rss_enabled=rss.get("enabled", False),
        rss_feeds=tuple(rss.get("feeds", [])),
        rss_limit_per_feed=int(rss.get("limit_per_feed", 50)),
        google_enabled=google.get("enabled", False),
        google_language=str(google.get("language", "en")),
        google_region=str(google.get("region", "US")),
        google_limit_per_company=int(google.get("limit_per_company", 5)),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed.strip().lower()


def _prepare_keywords(keywords: Sequence[str]) -> tuple[str, ...]:
    prepared: list[str] = []
    for kw in keywords:
        normalized = _normalize_for_match(kw)
        if not normalized or normalized in prepared:
            continue
        prepared.append(normalized)
    return tuple(prepared)


def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    prepared = _prepare_keywords(keywords)
    if not prepared:
        return False
    target = _normalize_for_match(text)
    if not target:
        return False
    return any(kw in target for kw in prepared)


def _has_excluded(text: str, keywords: KeywordConfig) -> bool:
    if not keywords.exclude:
        return False
    prepared_exclude = _prepare_keywords(keywords.exclude)
    if not prepared_exclude:
        return False
    target = _normalize_for_match(text)
    if not target:
        return False
    return any(ex_kw in target for ex_kw in prepared_exclude)


def _climate_score(match_count: int) -> float:
    if match_count >= 3:
        return 1.0
    if match_count == 2:
        return 0.85
    if match_count == 1:
        return 0.65
    return 0.0


def _enrich_events(df: pd.DataFrame, keywords: KeywordConfig) -> pd.DataFrame:
    if df.empty:
        return df

    include_keywords = _prepare_keywords(keywords.include)
    exclude_keywords = _prepare_keywords(keywords.exclude)

    def _evaluate(row: pd.Series) -> pd.Series:
        title = row.get("title", "") or ""
        snippet = row.get("text_snippet", "") or ""
        text = f"{title} {snippet}".strip()
        normalized = _normalize_for_match(text)

        if exclude_keywords and any(ex_kw in normalized for ex_kw in exclude_keywords):
            return pd.Series(
                {
                    "signal_strength": 0.0,
                    "climate_score": 0.0,
                    "sentiment_label": "excluded",
                    "sentiment_score": 0.0,
                }
            )

        match_count = sum(1 for kw in include_keywords if kw in normalized)
        relevance = _climate_score(match_count)

        sentiment_score, sentiment_label = analyze_sentiment(text)

        base_strength = float(row.get("signal_strength", 0.5) or 0.0)

        if relevance == 0.0:
            return pd.Series(
                {
                    "signal_strength": 0.0,
                    "climate_score": 0.0,
                    "sentiment_label": "irrelevant",
                    "sentiment_score": 0.0,
                }
            )

        adjusted_strength = base_strength * relevance

        if sentiment_label == "negative":
            adjusted_strength = -abs(adjusted_strength * (1.0 + abs(sentiment_score)))
        elif sentiment_label == "neutral":
            adjusted_strength = adjusted_strength * 0.5
        else:  # positive
            adjusted_strength = adjusted_strength * (1.0 + sentiment_score * 0.5)

        adjusted_strength = max(min(adjusted_strength, 1.0), -1.0)

        return pd.Series(
            {
                "signal_strength": adjusted_strength,
                "climate_score": relevance,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        )

    metrics = df.apply(_evaluate, axis=1)
    enriched = df.copy()
    enriched[["signal_strength", "climate_score", "sentiment_label", "sentiment_score"]] = metrics
    enriched = enriched[enriched["signal_strength"] != 0.0]
    return enriched


def _gdelt_query(company_name: str, domain: str | None, keywords: KeywordConfig) -> str:
    terms: List[str] = []
    if company_name:
        terms.append(f'"{company_name}"')
    if domain:
        terms.append(f'"{domain}"')
    company_clause = " OR ".join(terms)
    include_clause = " OR ".join(f'"{kw}"' for kw in keywords.include) if keywords.include else ""
    query_parts = []
    if company_clause:
        query_parts.append(f"({company_clause})")
    if include_clause:
        query_parts.append(f"({include_clause})")
    if not query_parts:
        return ""
    return " AND ".join(query_parts)


def _fetch_gdelt(
    *,
    company_name: str,
    domain: str | None,
    keywords: KeywordConfig,
    since: dt.datetime,
    max_records: int,
) -> pd.DataFrame:
    query = _gdelt_query(company_name, domain, keywords)
    if not query:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
        "startdatetime": since.strftime("%Y%m%d%H%M%S"),
    }
    try:
        response = requests.get(GDELT_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows: List[dict] = []
    for article in payload.get("articles", [])[:max_records]:
        url = article.get("url") or ""
        title = _normalize(article.get("title", ""))
        snippet = _normalize(article.get("snippet", ""))
        if not url or not title:
            continue
        seen = article.get("seendate")
        if seen:
            ts_dt = dt.datetime.strptime(seen, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
        else:
            ts_dt = dt.datetime.now(dt.timezone.utc)
        if ts_dt < since:
            continue
        strength = 1.0 if _contains_keyword(title + " " + snippet, keywords.include) else 0.5
        if _has_excluded(title + " " + snippet, keywords):
            continue
        rows.append({
            "url": url,
            "title": title,
            "text_snippet": snippet,
            "ts": ts_dt.isoformat(),
            "signal_strength": strength,
        })
    return pd.DataFrame(rows)


def _fetch_rss(feeds: Sequence[str], limit: int) -> dict[str, List[dict]]:
    """Return feed entries keyed by feed URL."""
    data: dict[str, List[dict]] = {}
    for feed_url in feeds:
        parsed = feedparser.parse(feed_url)
        entries: List[dict] = []
        for entry in parsed.entries[:limit]:
            url = getattr(entry, "link", "")
            title = _normalize(getattr(entry, "title", ""))
            snippet = _normalize(getattr(entry, "summary", ""))
            if not url or not title:
                continue
            ts_dt: dt.datetime
            ts_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if ts_struct:
                ts_dt = dt.datetime(*ts_struct[:6], tzinfo=dt.timezone.utc)
            else:
                ts_dt = dt.datetime.now(dt.timezone.utc)
            entries.append({
                "url": url,
                "title": title,
                "text_snippet": snippet,
                "ts_dt": ts_dt,
            })
        data[feed_url] = entries
    return data


def _match_rss_entries(
    rss_data: dict[str, List[dict]],
    *,
    company_name: str,
    keywords: KeywordConfig,
    since: dt.datetime,
) -> pd.DataFrame:
    if not rss_data:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    pattern = re.compile(re.escape(company_name), re.IGNORECASE)
    for entries in rss_data.values():
        for item in entries:
            text = f"{item['title']} {item['text_snippet']}"
            if not pattern.search(text):
                continue
            strength = 1.0 if _contains_keyword(text, keywords.include) else 0.5
            if _has_excluded(text, keywords):
                continue
            if item["ts_dt"] < since:
                continue
            rows.append({
                "url": item["url"],
                "title": item["title"],
                "text_snippet": item["text_snippet"],
                "ts": item["ts_dt"].isoformat(),
                "signal_strength": strength,
            })
    return pd.DataFrame(rows)


def _build_google_query(company_name: str, keywords: KeywordConfig, domain: str | None = None) -> str:
    terms = [f'"{company_name}"']
    if domain:
        terms.append(f'"{domain}"')
    if keywords.include:
        include_clause = " OR ".join(f'"{kw}"' for kw in keywords.include)
        terms.append(f"({include_clause})")
    return " ".join(terms)


def _fetch_google_news(
    *,
    company_name: str,
    keywords: KeywordConfig,
    language: str,
    region: str,
    limit: int,
    since: dt.datetime,
    domain: str | None = None,
) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    from urllib.parse import quote

    def _request_feed(query: str) -> List[dict]:
        q = quote(query)
        lang = language or "en"
        reg = region or "US"
        url = (
            "https://news.google.com/rss/search?q="
            + q
            + f"&hl={lang}&gl={reg}&ceid={reg}:{lang}"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception:
            return []
        parsed = feedparser.parse(response.content)
        entries: List[dict] = []
        for entry in parsed.entries[:limit]:
            url_entry = getattr(entry, "link", "")
            title = _normalize(getattr(entry, "title", ""))
            snippet = _normalize(getattr(entry, "summary", ""))
            if not url_entry or not title:
                continue
            ts_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if ts_struct:
                ts_dt = dt.datetime(*ts_struct[:6], tzinfo=dt.timezone.utc)
            else:
                ts_dt = dt.datetime.now(dt.timezone.utc)
            entries.append({
                "url": url_entry,
                "title": title,
                "snippet": snippet,
                "ts_dt": ts_dt,
            })
        return entries

    rows: List[dict] = []
    primary_query = _build_google_query(company_name, keywords, domain)
    if primary_query:
        for item in _request_feed(primary_query):
            if item["ts_dt"] < since:
                continue
            text = f"{item['title']} {item['snippet']}"
            if _has_excluded(text, keywords):
                continue
            strength = 1.0 if keywords.include and _contains_keyword(text, keywords.include) else 0.6
            rows.append({
                "url": item["url"],
                "title": item["title"],
                "text_snippet": item["snippet"],
                "ts": item["ts_dt"].isoformat(),
                "signal_strength": strength,
            })

    if not rows:
        fallback_terms = [f'"{company_name}"']
        if domain:
            fallback_terms.append(f'"{domain}"')
        fallback_query = " OR ".join(fallback_terms)
        for item in _request_feed(fallback_query):
            if item["ts_dt"] < since:
                continue
            rows.append({
                "url": item["url"],
                "title": item["title"],
                "text_snippet": item["snippet"],
                "ts": item["ts_dt"].isoformat(),
                "signal_strength": 0.4,
            })

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


def collect_news(
    *,
    universe_path: str = "data/processed/universe_sample.csv",
    keywords_path: str = "config/keywords.yml",
    aggregator_path: str = "config/news.yml",
    months: int = 12,
    since: str | None = None,
    max_companies: int = 0,
    country: str | None = None,
    industry: str | None = None,
) -> int:
    keywords = _load_keywords(keywords_path)
    providers = _load_providers(aggregator_path)
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
    if universe.empty:
        return 0

    rss_data = _fetch_rss(providers.rss_feeds, providers.rss_limit_per_feed) if providers.rss_enabled else {}

    frames: List[pd.DataFrame] = []
    for _, row in universe.iterrows():
        company_id = row["company_id"]
        company_name = row["company_name"]
        domain = row.get("company_domain", "")
        domain = domain if isinstance(domain, str) and domain else None

        company_frames: List[pd.DataFrame] = []
        if providers.gdelt_enabled:
            df_gdelt = _fetch_gdelt(
                company_name=company_name,
                domain=domain,
                keywords=keywords,
                since=since_dt,
                max_records=providers.gdelt_max_records,
            )
            if not df_gdelt.empty:
                company_frames.append(df_gdelt)

        if providers.rss_enabled and rss_data:
            df_rss = _match_rss_entries(
                rss_data,
                company_name=company_name,
                keywords=keywords,
                since=since_dt,
            )
            if not df_rss.empty:
                company_frames.append(df_rss)

        if providers.google_enabled:
            df_google = _fetch_google_news(
                company_name=company_name,
                keywords=keywords,
                language=providers.google_language,
                region=providers.google_region,
                limit=providers.google_limit_per_company,
                since=since_dt,
                domain=domain,
            )
            if not df_google.empty:
                company_frames.append(df_google)

        if not company_frames:
            continue

        combined = pd.concat(company_frames, ignore_index=True)
        combined["company_id"] = company_id
        combined["company_qid"] = row.get("company_qid", row.get("qid", ""))
        combined["company_name"] = company_name
        combined["country"] = row.get("country", "")
        combined["industry"] = row.get("industry", "")
        combined["size_bin"] = row.get("size_bin", "")
        combined["source"] = "news"
        combined["signal_type"] = "news"
        combined = _enrich_events(combined, keywords)
        if combined.empty:
            continue
        frames.append(combined[EVENT_COLUMNS])

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)
    raw_path = _persist_raw_news(
        combined[EVENT_COLUMNS], country=country, industry=industry
    )
    if raw_path is not None:
        print(f"[collect-news] dump crudo -> {raw_path}")
    return append_events(combined, source="news", signal_type="news")


def run_collect_news(**kwargs) -> int:
    """CLI shim for backwards compatibility."""
    return collect_news(**kwargs)
RAW_NEWS_DIR = Path("data/raw/news")


def _slug(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_") or fallback
    return cleaned[:40]


def _persist_raw_news(df: pd.DataFrame, *, country: str | None, industry: str | None) -> Path | None:
    if df.empty:
        return None
    RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    country_slug = _slug(country, "ALL")
    industry_slug = _slug(industry, "ALL")
    run_dir = RAW_NEWS_DIR / f"{country_slug}_{industry_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ts_label = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"{ts_label}_{len(df)}.csv"
    df.to_csv(path, index=False)
    return path
