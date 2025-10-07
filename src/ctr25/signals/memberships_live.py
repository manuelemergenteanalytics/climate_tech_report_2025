"""Live fetchers for membership lists (SBTi, RE100, UNGC, B-Corps)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser
import yaml

from ctr25.utils.http import get
from ctr25.utils.text import clean_text

COLUMNS = [
    "member_name",
    "url",
    "ts",
    "source",
    "signal_type",
    "country",
    "sector",
]

TARGET_ISO = {
    "argentina": "AR",
    "brazil": "BR",
    "brasil": "BR",
    "chile": "CL",
    "colombia": "CO",
    "mexico": "MX",
    "méxico": "MX",
    "peru": "PE",
    "perú": "PE",
    "uruguay": "UY",
    "paraguay": "PY",
    "bolivia": "BO",
}


def _to_iso(country: str) -> str:
    if not country:
        return ""
    key = country.strip().lower()
    return TARGET_ISO.get(key, "")


def _parse_date(value: str) -> str:
    value = clean_text(value)
    if not value:
        return ""
    try:
        dt = parser.parse(value, dayfirst=True, yearfirst=False)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


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


def _iter_rows(table, *, name_selector: str, skip_classes: Iterable[str] | None = None):
    skip_classes = set(skip_classes or [])
    rows = table.find_all("tr")
    idx = 0
    while idx < len(rows):
        row = rows[idx]
        row_classes = row.get("class", [])
        if any(cls in skip_classes for cls in row_classes):
            idx += 1
            continue
        name_cell = row.select_one(name_selector)
        if not name_cell:
            idx += 1
            continue
        yield row, name_cell
        idx += 1


def _fetch_re100_table(url: str, *, max_pages: int = 200) -> pd.DataFrame:
    records: list[dict] = []
    seen: set[str] = set()
    for page in range(max_pages):
        response = get(url, params={"page": page})
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        if not table:
            break
        found_main = False
        rows = table.find_all("tr")
        i = 1  # skip header row
        while i < len(rows):
            row = rows[i]
            name_cell = row.find("th", class_="notranslate")
            if not name_cell:
                i += 1
                continue
            found_main = True
            member = clean_text(name_cell.get_text())
            if not member or member.lower() in seen:
                i += 2
                continue
            seen.add(member.lower())
            cols = row.find_all("td")
            membership = clean_text(cols[0].get_text() if len(cols) >= 1 else "")
            join_year = clean_text(cols[1].get_text() if len(cols) >= 2 else "")
            target_year = clean_text(cols[2].get_text() if len(cols) >= 3 else "")
            industry = clean_text(cols[4].get_text() if len(cols) >= 5 else "")
            hq = clean_text(cols[-1].get_text() if cols else "")
            iso = _to_iso(hq)
            if not iso:
                i += 2
                continue
            link = name_cell.find("a")
            href = link.get("href") if link else url
            if href and href.startswith("/"):
                href = urljoin(url, href)
            join_iso = f"{join_year}-01-01" if join_year.isdigit() else ""
            records.append(
                {
                    "member_name": member,
                    "url": href or url,
                    "ts": join_iso,
                    "source": "memberships",
                    "signal_type": "re100",
                    "country": iso,
                    "sector": industry,
                }
            )
            i += 2  # skip description row
        if not found_main:
            break
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(records)


def _fetch_ungc_table(url: str, *, max_pages: int = 40) -> pd.DataFrame:
    records: list[dict] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        response = get(url, params={"page": page})
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        if not table:
            break
        rows = table.find_all("tr")
        found_any = False
        for row in rows[1:]:  # skip header
            name_cell = row.find("th", class_="name")
            if not name_cell:
                continue
            found_any = True
            member = clean_text(name_cell.get_text())
            if not member or member.lower() in seen:
                continue
            country_cell = row.find("td", class_="country")
            country_name = clean_text(country_cell.get_text() if country_cell else "")
            iso = _to_iso(country_name)
            if not iso:
                continue
            join_cell = row.find("td", class_="joined-on")
            joined = _parse_date(join_cell.get_text() if join_cell else "")
            type_cell = row.find("td", class_="type")
            sector_cell = row.find("td", class_="sector")
            link = name_cell.find("a")
            href = link.get("href") if link else url
            if href and href.startswith("/"):
                href = urljoin(url, href)
            records.append(
                {
                    "member_name": member,
                    "url": href or url,
                    "ts": joined,
                    "source": "memberships",
                    "signal_type": "ungc",
                    "country": iso,
                    "sector": clean_text(sector_cell.get_text() if sector_cell else ""),
                }
            )
            seen.add(member.lower())
        if not found_any:
            break
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(records)


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
    if "there100" in list_url:
        return _fetch_re100_table(list_url)
    return _fetch(list_url, "re100")


def fetch_ungc(list_url: str) -> pd.DataFrame:
    if "unglobalcompact" in list_url:
        return _fetch_ungc_table(list_url)
    return _fetch(list_url, "ungc")


def fetch_bcorps(list_url: str) -> pd.DataFrame:
    return _fetch(list_url, "bcorps")


def load_memberships_cfg(path: str = "config/memberships.yml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
