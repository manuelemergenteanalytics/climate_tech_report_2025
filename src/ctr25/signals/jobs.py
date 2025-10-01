"""Jobs collectors using Greenhouse and Lever APIs."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

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


def _parse_epoch_ms(value) -> str:
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value / 1000.0, tz=dt.timezone.utc).isoformat()
    return _utc_now_iso()


def _make_event(*, url: str, title: str, snippet: str, ts: str) -> dict:
    return {
        "company_id": "",
        "company_name": "",
        "country": "",
        "industry": "",
        "size_bin": "",
        "source": "jobs",
        "signal_type": "job",
        "signal_strength": 1.0,
        "ts": ts,
        "url": url,
        "title": title[:300],
        "text_snippet": snippet[:500],
    }


def _extract_domain(domain_or_url: str) -> str:
    if not domain_or_url:
        return ""
    if "://" in domain_or_url:
        return urlparse(domain_or_url).netloc
    return domain_or_url


def fetch_greenhouse(board_token: str) -> pd.DataFrame:
    token = board_token.strip()
    if not token:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    try:
        payload = get(url).json()
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for job in payload.get("jobs", []):
        job_url = job.get("absolute_url") or ""
        if not job_url:
            continue
        title = job.get("title") or ""
        location = (job.get("location") or {}).get("name", "")
        ts = job.get("updated_at") or job.get("created_at") or _utc_now_iso()
        rows.append(_make_event(url=job_url, title=title, snippet=location, ts=ts))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def fetch_lever(company: str) -> pd.DataFrame:
    slug = company.strip()
    if not slug:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    url = f"https://api.lever.co/v0/postings/{slug}"
    try:
        payload = get(url, params={"mode": "json"}).json()
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for job in payload:
        job_url = job.get("hostedUrl") or ""
        if not job_url:
            continue
        title = job.get("text") or job.get("name") or ""
        categories = job.get("categories", {}) or {}
        snippet = ", ".join(str(v) for v in categories.values() if v)
        ts = _parse_epoch_ms(job.get("createdAt"))
        rows.append(_make_event(url=job_url, title=title, snippet=snippet, ts=ts))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _load_jobs_cfg(path: str) -> dict:
    default = {"greenhouse_boards": [], "lever_companies": []}
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {
        "greenhouse_boards": data.get("greenhouse_boards", []),
        "lever_companies": data.get("lever_companies", []),
    }


def run_collect_jobs(
    universe_path: str = "data/processed/universe_sample.csv",
    keywords_path: str = "config/keywords.yml",
    country: str | None = None,
    industry: str | None = None,
    max_companies: int = 0,
    jobs_cfg_path: str = "config/jobs.yml",
) -> int:
    cfg = _load_jobs_cfg(jobs_cfg_path)
    frames: list[pd.DataFrame] = []

    for board in cfg.get("greenhouse_boards", []):
        frames.append(fetch_greenhouse(board))

    for company in cfg.get("lever_companies", []):
        frames.append(fetch_lever(company))

    frames = [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return 0

    added = append_multiple(frames)
    return added
