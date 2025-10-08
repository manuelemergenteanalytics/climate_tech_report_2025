"""Collect job postings from Greenhouse and Lever boards."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import requests
import requests_cache
import yaml

from ctr25.utils.events import append_events, resolve_since
from ctr25.utils.universe import UniverseFilters, load_universe

_CACHE_PATH = Path("data/interim/cache/jobs_requests.sqlite")
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
]


@dataclass
class BoardConfig:
    slug: str
    company_domain: str


@dataclass
class JobsConfig:
    greenhouse: Sequence[BoardConfig]
    lever: Sequence[BoardConfig]


def _load_jobs_cfg(path: str) -> JobsConfig:
    p = Path(path)
    if not p.exists():
        return JobsConfig(greenhouse=(), lever=())
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    def _parse_boards(entries: Iterable[dict]) -> List[BoardConfig]:
        boards: List[BoardConfig] = []
        for entry in entries or []:
            slug = str(entry.get("slug", "")).strip()
            domain = str(entry.get("company_domain", "")).strip()
            if slug and domain:
                boards.append(BoardConfig(slug=slug, company_domain=domain))
        return boards

    return JobsConfig(
        greenhouse=_parse_boards(raw.get("greenhouse", [])),
        lever=_parse_boards(raw.get("lever", [])),
    )


def _fetch_greenhouse(slug: str) -> pd.DataFrame:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        payload = requests.get(url, timeout=30).json()
    except Exception:
        return pd.DataFrame(columns=["title", "url", "snippet", "ts"])
    rows = []
    for job in payload.get("jobs", []):
        job_url = job.get("absolute_url")
        title = job.get("title")
        location = (job.get("location") or {}).get("name", "")
        ts = job.get("updated_at") or job.get("created_at") or dt.datetime.now(dt.timezone.utc).isoformat()
        if job_url and title:
            rows.append({"title": title, "url": job_url, "snippet": location, "ts": ts})
    return pd.DataFrame(rows)


def _fetch_lever(slug: str) -> pd.DataFrame:
    url = f"https://api.lever.co/v0/postings/{slug}"
    try:
        payload = requests.get(url, params={"mode": "json"}, timeout=30).json()
    except Exception:
        return pd.DataFrame(columns=["title", "url", "snippet", "ts"])
    rows = []
    for job in payload:
        job_url = job.get("hostedUrl")
        title = job.get("text") or job.get("name")
        categories = job.get("categories", {}) or {}
        snippet = ", ".join(str(v) for v in categories.values() if v)
        ts = job.get("createdAt")
        if isinstance(ts, (int, float)):
            ts_dt = dt.datetime.fromtimestamp(ts / 1000.0, tz=dt.timezone.utc)
            ts_iso = ts_dt.isoformat()
        else:
            ts_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        if job_url and title:
            rows.append({"title": title, "url": job_url, "snippet": snippet, "ts": ts_iso})
    return pd.DataFrame(rows)


def _strength_from_keywords(text: str, keywords: Sequence[str]) -> float:
    lowered = text.lower()
    for kw in keywords:
        if kw.lower() in lowered:
            return 1.0
    return 0.5


def collect_jobs(
    *,
    universe_path: str = "data/processed/universe_sample.csv",
    jobs_cfg_path: str = "config/jobs.yml",
    keywords_path: str = "config/keywords.yml",
    months: int = 12,
    since: str | None = None,
    max_companies: int = 0,
    country: str | None = None,
    industry: str | None = None,
) -> int:
    cfg = _load_jobs_cfg(jobs_cfg_path)
    if not cfg.greenhouse and not cfg.lever:
        return 0

    with Path(keywords_path).open("r", encoding="utf-8") as fh:
        kw_data = yaml.safe_load(fh) or {}
    keywords = tuple(map(str, kw_data.get("include", [])))

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

    universe_by_domain = {
        str(row["company_domain"]).lower(): row
        for _, row in universe.iterrows()
        if isinstance(row.get("company_domain"), str) and row.get("company_domain")
    }

    frames: List[pd.DataFrame] = []

    def _materialize(board: BoardConfig, fetch_fn) -> None:
        company = universe_by_domain.get(board.company_domain.lower())
        if company is None:
            return
        df = fetch_fn(board.slug)
        if df.empty:
            return
        df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df[df["ts_dt"] >= since_dt]
        if df.empty:
            return
        df["signal_strength"] = df.apply(
            lambda r: _strength_from_keywords(
                f"{r['title']} {r['snippet']}", keywords
            ),
            axis=1,
        )
        df["company_id"] = company["company_id"]
        df["company_qid"] = company.get("company_qid", company.get("qid", ""))
        df["company_name"] = company["company_name"]
        df["country"] = company.get("country", "")
        df["industry"] = company.get("industry", "")
        df["size_bin"] = company.get("size_bin", "")
        df["source"] = "jobs"
        df["signal_type"] = "job_posting"
        df.rename(columns={"snippet": "text_snippet"}, inplace=True)
        df["ts"] = df["ts_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frames.append(df[EVENT_COLUMNS])

    for board in cfg.greenhouse:
        _materialize(board, _fetch_greenhouse)

    for board in cfg.lever:
        _materialize(board, _fetch_lever)

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)
    return append_events(combined, source="jobs", signal_type="job_posting")


def run_collect_jobs(**kwargs) -> int:
    return collect_jobs(**kwargs)
