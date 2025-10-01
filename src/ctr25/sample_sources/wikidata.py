"""Utilities to build the LATAM universe from Wikidata."""
from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import requests_cache

default_cfg_mappings = {
    "mappings": [
        {"pattern": "(?i)oil|petro|gas|hidrocarb", "to": "oil_gas"},
        {"pattern": "(?i)energy|energ[ií]a|eléctr|electric|power", "to": "energy_power"},
        {"pattern": "(?i)mining|min[ií]er|minería|minera", "to": "mining_metals"},
        {"pattern": "(?i)chem|qu[ií]m|material", "to": "chemicals_materials"},
        {"pattern": "(?i)manufactur|fábric|industrial", "to": "manufacturing"},
        {"pattern": "(?i)construct|construcci[oó]n|real estate|inmobili", "to": "construction_realestate"},
        {"pattern": "(?i)transport|logist|ferro|a[eé]re|aero|aviaci|metro", "to": "transport_logistics"},
        {"pattern": "(?i)agri|agro|food|alimento", "to": "agro_food"},
        {"pattern": "(?i)retail|consumer|consumo|comercio", "to": "retail_consumer"},
        {"pattern": "(?i)water|waste|residu|circular|h[ií]dric", "to": "water_waste_circularity"},
        {"pattern": "(?i)finance|bank|banca|financ|insur|seguro", "to": "finance_insurance"},
        {"pattern": "(?i)ict|telecom|telefon|software|internet|tech|tecnolog", "to": "ict_telecom"},
    ]
}

ENDPOINT = "https://query.wikidata.org/sparql"
COUNTRY_QIDS: Dict[str, str] = {
    "MX": "wd:Q96",
    "BR": "wd:Q155",
    "CO": "wd:Q739",
    "CL": "wd:Q298",
    "AR": "wd:Q414",
    "UY": "wd:Q77",
}

_CACHE_PATH = Path("data/interim/cache/wikidata.sqlite")
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
requests_cache.install_cache(str(_CACHE_PATH), backend="sqlite", expire_after=3600)


def _run_sparql(query: str) -> dict:
    headers = {
        "User-Agent": "ctr25/1.0 (Wikidata sampling)",
        "Accept": "application/sparql-results+json",
    }
    for attempt in range(3):
        try:
            response = requests.get(
                ENDPOINT,
                params={"format": "json", "query": query},
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def _domain_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url.strip())
    host = parsed.netloc or parsed.path
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split("/")[0]


def _size_bin(employees: float) -> str:
    if employees >= 1000:
        return "l"
    if employees >= 250:
        return "m"
    if employees >= 50:
        return "s"
    return ""


def _hash_company_id(qid: str) -> int:
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


SPARQL_TEMPLATE = """
SELECT ?company ?companyLabel ?countryCode ?industryLabel ?employees ?website ?ticker ?exchangeLabel WHERE {{
  VALUES ?targetCountry {{ {country_qid} }}
  ?company wdt:P31/wdt:P279* wd:Q4830453.
  {location_block}
  ?company {employee_property} ?employees.
  OPTIONAL {{ ?company wdt:P856 ?website. }}
  OPTIONAL {{ ?company wdt:P452 ?industry. }}
  OPTIONAL {{
    ?company p:P414 ?exchangeStmt.
    ?exchangeStmt ps:P414 ?exchange;
                   pq:P249 ?ticker.
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
}}
LIMIT {limit}
OFFSET {offset}
"""


def query_wikidata(companies_per_country: int = 500) -> pd.DataFrame:
    records: Dict[str, Dict[str, object]] = {}
    country_records: Dict[str, set] = {code: set() for code in COUNTRY_QIDS}

    chunk = max(25, min(80, int(companies_per_country)))

    location_blocks = [
        """
  ?company wdt:P17 ?targetCountry.
  BIND(?targetCountry AS ?countryEntity)
  ?countryEntity wdt:P297 ?countryCode.
""",
        """
  ?company wdt:P159 ?hq.
  ?hq wdt:P17 ?targetCountry.
  BIND(?targetCountry AS ?countryEntity)
  ?countryEntity wdt:P297 ?countryCode.
""",
    ]

    employee_properties = ["wdt:P1128", "wdt:P1081"]

    for loop_country_code, country_qid in COUNTRY_QIDS.items():
        for location_block in location_blocks:
            if len(country_records[loop_country_code]) >= companies_per_country:
                break
            for employee_property in employee_properties:
                if len(country_records[loop_country_code]) >= companies_per_country:
                    break
                offset = 0
                while len(country_records[loop_country_code]) < companies_per_country:
                    remaining = companies_per_country - len(country_records[loop_country_code])
                    limit = min(chunk, remaining)
                    query = SPARQL_TEMPLATE.format(
                        country_qid=country_qid,
                        location_block=location_block,
                        employee_property=employee_property,
                        limit=int(limit),
                        offset=int(offset),
                    )
                    try:
                        payload = _run_sparql(query)
                    except requests.exceptions.RequestException:
                        break
                    bindings = payload.get("results", {}).get("bindings", [])
                    if not bindings:
                        break
                    offset += limit
                    for item in bindings:
                        uri = item.get("company", {}).get("value")
                        if not uri:
                            continue
                        qid = uri.rsplit("/", 1)[-1]
                        country_code = item.get("countryCode", {}).get("value", loop_country_code)
                        country_code = country_code.upper() if isinstance(country_code, str) else loop_country_code
                        rec = records.setdefault(
                            qid,
                            {
                                "qid": qid,
                                "company_name": item.get("companyLabel", {}).get("value", "").strip(),
                                "country": country_code,
                                "industry_labels": set(),
                                "employees": None,
                                "company_domain": "",
                                "tickers": set(),
                            },
                        )

                        industry_label = item.get("industryLabel", {}).get("value")
                        if industry_label:
                            rec["industry_labels"].add(industry_label.strip())

                        employees_value = item.get("employees", {}).get("value")
                        if employees_value:
                            try:
                                employees = float(employees_value)
                            except (TypeError, ValueError):
                                employees = None
                            if employees is not None:
                                current = rec.get("employees") or 0.0
                                if employees > current:
                                    rec["employees"] = employees

                        website = item.get("website", {}).get("value")
                        if website:
                            domain = _domain_from_url(website)
                            if domain:
                                rec["company_domain"] = domain

                        ticker_value = item.get("ticker", {}).get("value")
                        if ticker_value:
                            exchange_label = item.get("exchangeLabel", {}).get("value", "").strip()
                            ticker = ticker_value.strip()
                            if exchange_label:
                                rec["tickers"].add(f"{ticker}@{exchange_label}")
                            else:
                                rec["tickers"].add(ticker)

                        country_records[loop_country_code].add(qid)

                    if len(bindings) < limit:
                        break
                    time.sleep(0.5)

    rows: List[Dict[str, object]] = []
    for data in records.values():
        company_name = data.get("company_name", "").strip()
        employees = data.get("employees")
        if not company_name or employees is None:
            continue
        size = _size_bin(employees)
        if not size:
            continue
        country_code = data.get("country", "")
        if country_code not in COUNTRY_QIDS:
            continue
        industries = sorted(data.get("industry_labels", []))
        if not industries:
            continue
        domain = data.get("company_domain", "")
        tickers = sorted(data.get("tickers", []))
        if not domain and not tickers:
            continue
        rows.append(
            {
                "qid": data["qid"],
                "company_name": company_name,
                "country": country_code,
                "industry_raw": "; ".join(industries),
                "employees": employees,
                "company_domain": domain,
                "ticker": tickers[0] if tickers else "",
                "size_bin": size,
            }
        )

    return pd.DataFrame(rows)


def map_industries(df: pd.DataFrame, map_path: str = "config/industry_map.yml") -> pd.DataFrame:
    if df.empty:
        return df

    path = Path(map_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(default_cfg_mappings, fh, sort_keys=False, allow_unicode=False)

    import yaml

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    mappings = cfg.get("mappings")
    if not mappings:
        legacy: List[Dict[str, object]] = []
        for slug, aliases in cfg.items():
            if not isinstance(aliases, Iterable):
                continue
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                pattern = rf"(?i){re.escape(alias.strip())}"
                legacy.append({"pattern": pattern, "to": slug})
        mappings = legacy
    patterns: List[Tuple[re.Pattern, str]] = []
    for entry in mappings:
        pattern = entry.get("pattern")
        target = entry.get("to")
        if not pattern or not target:
            continue
        patterns.append((re.compile(pattern), target))

    def _map(row: pd.Series) -> str:
        raw = str(row.get("industry_raw", ""))
        for rx, target in patterns:
            if rx.search(raw):
                return target
        return ""

    df = df.copy()
    df["industry"] = df.apply(_map, axis=1)
    df = df[df["industry"].astype(bool)].copy()
    return df


def apply_sampling(df: pd.DataFrame, project_cfg: Dict[str, object]) -> pd.DataFrame:
    sample_cfg = (project_cfg or {}).get("sample", project_cfg or {})
    countries = sample_cfg.get("countries")
    industries = sample_cfg.get("industries")
    min_per_stratum = int(sample_cfg.get("min_per_stratum", 20))
    total_target = int(sample_cfg.get("total_target", 1200))

    if countries:
        df = df[df["country"].isin(countries)].copy()
    if industries:
        df = df[df["industry"].isin(industries)].copy()

    df = df[df["size_bin"].isin(["s", "m", "l"])]
    df = df.drop_duplicates(subset=["qid"])
    df = df.sort_values(["country", "industry", "employees"], ascending=[True, True, False])

    strata = []
    if countries and industries:
        for country in countries:
            for industry in industries:
                mask = (df["country"] == country) & (df["industry"] == industry)
                slice_df = df[mask]
                if not slice_df.empty:
                    strata.append((country, industry, slice_df))
    else:
        for key, slice_df in df.groupby(["country", "industry"], sort=False):
            strata.append((key[0], key[1], slice_df))

    if not strata:
        return pd.DataFrame(columns=[
            "company_id",
            "company_name",
            "country",
            "industry",
            "size_bin",
            "company_domain",
            "weight_stratum",
            "ticker",
        ])

    n_strata = len(strata)
    target_per_stratum = max(min_per_stratum, math.floor(total_target / n_strata))
    sampled_frames: List[pd.DataFrame] = []
    stratum_sizes: Dict[Tuple[str, str], int] = {}

    for country, industry, slice_df in strata:
        take = min(len(slice_df), target_per_stratum)
        if take <= 0:
            continue
        sampled = slice_df.head(take).copy()
        sampled_frames.append(sampled)
        stratum_sizes[(country, industry)] = len(sampled)

    if not sampled_frames:
        return pd.DataFrame(columns=[
            "company_id",
            "company_name",
            "country",
            "industry",
            "size_bin",
            "company_domain",
            "weight_stratum",
            "ticker",
        ])

    sample_df = pd.concat(sampled_frames, ignore_index=True)
    sample_df = sample_df.sort_values(["country", "industry", "company_name"]).reset_index(drop=True)

    weight_map: Dict[Tuple[str, str], float] = {}
    for key, count in stratum_sizes.items():
        if count <= 0:
            continue
        weight_map[key] = total_target / (n_strata * count)

    sample_df["weight_stratum"] = sample_df.apply(
        lambda row: weight_map.get((row["country"], row["industry"]), 1.0),
        axis=1,
    )

    ordered_qids = sorted(sample_df["qid"].tolist())
    qid_to_id = {qid: idx + 1 for idx, qid in enumerate(ordered_qids)}
    sample_df["company_id"] = sample_df["qid"].map(qid_to_id)

    columns = [
        "company_id",
        "company_name",
        "country",
        "industry",
        "size_bin",
        "company_domain",
        "weight_stratum",
        "ticker",
    ]

    result = sample_df[columns].copy().sort_values("company_id").reset_index(drop=True)
    return result
