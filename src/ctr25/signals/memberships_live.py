"""Live fetchers for membership lists (SBTi, RE100, UNGC, B-Corps)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
import yaml

from ctr25.utils.http import get

COLUMNS = [
    "member_name",
    "url",
    "ts",
    "source",
    "signal_type",
    "country",
    "sector",
]


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
        df = pd.read_csv(url, dtype=str)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    lower_cols = {c.lower(): c for c in df.columns}
    name_col = None
    for key in (
        "company_name",
        "company",
        "organization",
        "organizacion",
        "organización",
        "nome",
        "nombre",
    ):
        if key in lower_cols:
            name_col = lower_cols[key]
            break
    if not name_col:
        name_col = df.columns[0]
    ts_col = None
    for key in (
        "date_updated",
        "date",
        "fecha",
        "joined",
        "approved",
        "commitment",
    ):
        if key in lower_cols:
            ts_col = lower_cols[key]
            break
    country_col = None
    for key in ("country", "location", "pais", "país"):
        if key in lower_cols:
            country_col = lower_cols[key]
            break
    sector_col = None
    for key in ("sector", "industry", "categoria"):
        if key in lower_cols:
            sector_col = lower_cols[key]
            break
    records = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, ""))
        if not name.strip():
            continue
        ts_value = str(row.get(ts_col, "")) if ts_col else ""
        record = _mk_record(name=name, url=url, ts=ts_value, kind=kind)
        if country_col:
            record["country"] = row.get(country_col, "")
        if sector_col:
            record["sector"] = row.get(sector_col, "")
        records.append(record)
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame.from_records(records)


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
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(url, dtype=str)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
        lower_cols = {c.lower(): c for c in df.columns}
        name_col = (
            lower_cols.get("company_name")
            or lower_cols.get("company")
            or lower_cols.get("organization")
            or lower_cols.get("organización")
            or lower_cols.get("organisação")
            or df.columns[0]
        )
        ts_col = (
            lower_cols.get("date_updated")
            or lower_cols.get("date")
            or lower_cols.get("joined")
            or lower_cols.get("commitment")
            or lower_cols.get("approved")
            or lower_cols.get("fecha")
        )
        country_col = lower_cols.get("location") or lower_cols.get("country")
        sector_col = lower_cols.get("sector")
        records = []
        for _, row in df.iterrows():
            name = str(row.get(name_col, ""))
            if not name.strip():
                continue
            ts_value = str(row.get(ts_col, "")) if ts_col else ""
            record = _mk_record(name=name, url=url, ts=ts_value, kind=kind)
            if country_col:
                record["country"] = row.get(country_col, "")
            if sector_col:
                record["sector"] = row.get(sector_col, "")
            records.append(record)
        if not records:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame.from_records(records)
    try:
        df_csv = _from_csv(url, kind)
    except Exception:
        df_csv = pd.DataFrame(columns=COLUMNS)
    if not df_csv.empty:
        return df_csv
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
