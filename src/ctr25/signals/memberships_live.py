"""Live fetchers for membership lists (SBTi, RE100, UNGC, B-Corps)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from bs4 import BeautifulSoup
import yaml

from ctr25.utils.http import get

COLUMNS = ["member_name", "url", "ts", "source", "signal_type"]


def _mk_record(name: str, url: str, ts: str, kind: str) -> dict:
    return {
        "member_name": name.strip(),
        "url": url,
        "ts": ts or "",
        "source": "memberships",
        "signal_type": kind,
    }


def _from_csv(url: str, kind: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    lower_cols = {c.lower(): c for c in df.columns}
    name_col = None
    for key in ("company", "organization", "organizacion", "nombre"):
        if key in lower_cols:
            name_col = lower_cols[key]
            break
    if not name_col:
        name_col = df.columns[0]
    ts_col = None
    for key in ("date", "fecha", "joined", "approved", "commitment"):
        if key in lower_cols:
            ts_col = lower_cols[key]
            break
    records = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, ""))
        if not name.strip():
            continue
        ts_value = str(row.get(ts_col, "")) if ts_col else ""
        records.append(_mk_record(name=name, url=url, ts=ts_value, kind=kind))
    return pd.DataFrame(records, columns=COLUMNS)


def _from_html(url: str, kind: str) -> pd.DataFrame:
    response = get(url)
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
    if not table:
        return pd.DataFrame(columns=COLUMNS)
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if headers and len(headers) == len(cells):
            record = dict(zip(headers, cells))
        else:
            record = {f"col_{i}": cell for i, cell in enumerate(cells)}
        name = record.get("Company") or record.get("Organization") or record.get("Name") or next(iter(record.values()), "")
        if not name:
            continue
        ts = record.get("Date") or record.get("Joined") or record.get("Fecha") or ""
        rows.append(_mk_record(name=name, url=url, ts=ts, kind=kind))
    return pd.DataFrame(rows, columns=COLUMNS)


def _fetch(url: str, kind: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame(columns=COLUMNS)
    try:
        return _from_csv(url, kind)
    except Exception:
        return _from_html(url, kind)


def fetch_sbti(list_url: str) -> pd.DataFrame:
    return _fetch(list_url, "sbti")


def fetch_re100(list_url: str) -> pd.DataFrame:
    return _fetch(list_url, "re100")


def fetch_ungc(list_url: str) -> pd.DataFrame:
    return _fetch(list_url, "ungc")


def fetch_bcorps(list_url: str) -> pd.DataFrame:
    return _fetch(list_url, "bcorps")


def load_memberships_cfg(path: str = "config/memberships.yml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
