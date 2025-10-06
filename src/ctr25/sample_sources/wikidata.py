"""Utilities to build the LATAM universe from Wikidata."""
from __future__ import annotations

import hashlib
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import requests_cache
import yaml
from threading import local

# --- Default industry regex mappings (fallback file is created if missing) ---
default_cfg_mappings = {
    "mappings": [
        {"pattern": r"(?i)oil|petro|gas|hidrocarb|upstream|downstream", "to": "oil_gas"},
        {"pattern": r"(?i)energy|energ[ií]a|eléctr|electric|power|solar|fotovoltaic|e[oó]lic|wind|hidroel[eé]ctric|geoterm|renew|bioenerg|biofuel", "to": "energy_power"},
        {"pattern": r"(?i)mining|min[ií]er|minería|minera", "to": "mining_metals"},
        {"pattern": r"(?i)chem|qu[ií]m|material|siderurg|metalúrg|fertiliz|plástic|petroqu[ií]m|pol[ií]mer", "to": "chemicals_materials"},
        {"pattern": r"(?i)manufactur|fábric|industrial|automotriz|automotive|textil|maquil|ensambl", "to": "manufacturing"},
        {"pattern": r"(?i)construct|construcci[oó]n|real estate|inmobili|infraestruct|desarroll|viviend|obra", "to": "construction_realestate"},
        {"pattern": r"(?i)transport|logist|ferro|a[eé]re|aero|aviaci|metro|naval|shipping|turism|turíst|portu|mar[ií]t|aerol[ií]n", "to": "transport_logistics"},
        {"pattern": r"(?i)agri|agro|food|alimento|lácteo|bev|cervec|brew|helad|c[aá]fe|cacao|ganad|harin|az[ií]car", "to": "agro_food"},
        {"pattern": r"(?i)retail|consumer|consumo|comercio|minoris|tienda|supermerc|farmac|ferreter|cosm[eé]t|perfume|hogar|departamental|e-?commerce", "to": "retail_consumer"},
        {"pattern": r"(?i)water|waste|residu|circular|h[ií]dric|saneam|sanit|alcantarill|tratamiento", "to": "water_waste_circularity"},
        {"pattern": r"(?i)finance|bank|banca|financ|insur|seguro|holding|banco|fond[oa]|inversion|burs[aá]til|cooperativ|microfin", "to": "finance_insurance"},
        {"pattern": r"(?i)ict|telecom|telefon|software|internet|tech|tecnolog|medios|cine|animaci|televis|notici|audiovisual|videojueg|stream|rad[ií]o|digital|m[oó]vil", "to": "ict_telecom"},
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
    "PE": "wd:Q419",
    "EC": "wd:Q736",
    "BO": "wd:Q750",
    "PY": "wd:Q733",
    "PA": "wd:Q804",
    "DO": "wd:Q786",
    "VE": "wd:Q717",
    "GT": "wd:Q774",
    "SV": "wd:Q792",
    "HN": "wd:Q783",
    "NI": "wd:Q811",
    "CR": "wd:Q800",
}

# Cache HTTP 1h
_CACHE_PATH = Path("data/interim/cache/wikidata.sqlite")
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
_CACHE_BASENAME = str(_CACHE_PATH.with_suffix(""))
_THREAD_LOCAL = local()


def _get_session() -> requests_cache.CachedSession:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests_cache.CachedSession(
            cache_name=_CACHE_BASENAME,
            backend="sqlite",
            expire_after=3600,
        )
        _THREAD_LOCAL.session = session
    return session


def _run_sparql(query: str) -> dict:
    headers = {
        "User-Agent": "ctr25/1.0 (Wikidata sampling)",
        "Accept": "application/sparql-results+json",
    }
    for attempt in range(3):
        try:
            session = _get_session()
            r = session.get(
                ENDPOINT,
                params={"format": "json", "query": query},
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
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


def _normalize_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.,-]", "", value.strip()).replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _size_bin(employees: float | None, revenue: float | None, ticker: str | None) -> str:
    if employees is not None:
        if employees >= 1000:
            return "l"
        if employees >= 250:
            return "m"
        if employees >= 50:
            return "s"
    if revenue is not None:
        if revenue >= 1_000_000_000:
            return "l"
        if revenue >= 100_000_000:
            return "m"
        if revenue >= 10_000_000:
            return "s"
    if ticker:
        return "l"
    return "m"


def _hash_company_id(qid: str) -> int:
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


SPARQL_TEMPLATE = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT DISTINCT ?company ?companyLabel ?companyDescription ?industryLabel ?employees1128 ?employees1081 ?revenue ?website ?ticker ?exchangeLabel WHERE {{
  VALUES ?targetCountry {{ {country_qid} }}
  ?company wdt:P31/wdt:P279* wd:Q4830453 .
  {{
    ?company wdt:P17 ?targetCountry .
  }} UNION {{
    ?company wdt:P159 ?hq . ?hq wdt:P17 ?targetCountry .
  }}
  OPTIONAL {{ ?company wdt:P1128 ?employees1128. }}
  OPTIONAL {{ ?company wdt:P1081 ?employees1081. }}
  OPTIONAL {{ ?company wdt:P2139 ?revenue. }}
  OPTIONAL {{ ?company wdt:P856 ?website. }}
  OPTIONAL {{ ?company wdt:P452 ?industry. }}
  OPTIONAL {{
    ?company p:P414 ?exchangeStmt.
    ?exchangeStmt ps:P414 ?exchange ;
                   pq:P249 ?ticker .
    OPTIONAL {{ ?exchange rdfs:label ?exchangeLabel FILTER (lang(?exchangeLabel) = "en") }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en,pt". }}
}}
LIMIT {limit}
OFFSET {offset}
"""


def query_wikidata(
    companies_per_country: int = 500,
    countries: Iterable[str] | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    target = max(1, int(companies_per_country))
    chunk = max(25, min(50, max(25, target // 4)))

    if countries:
        country_codes: List[str] = []
        for code in countries:
            if not isinstance(code, str):
                continue
            iso = code.strip().upper()
            if iso in COUNTRY_QIDS:
                country_codes.append(iso)
        if not country_codes:
            raise ValueError("No se definieron países válidos para el muestreo.")
    else:
        country_codes = list(COUNTRY_QIDS.keys())

    def _fetch_country(iso: str) -> Dict[str, Dict[str, object]]:
        qid_country = COUNTRY_QIDS[iso]
        records: Dict[str, Dict[str, object]] = {}
        seen: set[str] = set()
        offset = 0

        while len(seen) < target:
            remaining = target - len(seen)
            limit = min(chunk, remaining)
            query = SPARQL_TEMPLATE.format(
                country_qid=qid_country,
                limit=int(limit),
                offset=int(offset),
            )
            try:
                payload = _run_sparql(query)
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"SPARQL falló para {iso} offset {offset}: {exc}") from exc

            bindings = payload.get("results", {}).get("bindings", [])
            if not bindings:
                break

            offset += limit

            for b in bindings:
                uri = b.get("company", {}).get("value")
                if not uri:
                    continue
                qid_item = uri.rsplit("/", 1)[-1]

                rec = records.setdefault(
                    qid_item,
                    {
                        "qid": qid_item,
                        "company_name": b.get("companyLabel", {}).get("value", "").strip(),
                        "country": iso,
                        "industry_labels": set(),
                        "description": b.get("companyDescription", {}).get("value", "").strip(),
                        "employees": None,
                        "revenue": None,
                        "company_domain": "",
                        "tickers": set(),
                    },
                )

                ind = b.get("industryLabel", {}).get("value")
                if ind:
                    rec["industry_labels"].add(ind.strip())

                e1 = _normalize_numeric(b.get("employees1128", {}).get("value"))
                e2 = _normalize_numeric(b.get("employees1081", {}).get("value"))
                if e1 is not None or e2 is not None:
                    cand = max([v for v in (e1, e2) if v is not None])
                    rec["employees"] = max(rec.get("employees") or 0.0, cand)

                rev = _normalize_numeric(b.get("revenue", {}).get("value"))
                if rev is not None:
                    rec["revenue"] = max(rec.get("revenue") or 0.0, rev)

                web = b.get("website", {}).get("value")
                if web:
                    dom = _domain_from_url(web)
                    if dom:
                        rec["company_domain"] = dom

                tic = b.get("ticker", {}).get("value")
                if tic:
                    exch = b.get("exchangeLabel", {}).get("value", "").strip()
                    rec["tickers"].add(f"{tic}@{exch}" if exch else tic)

                seen.add(qid_item)

            if len(bindings) < limit:
                break
            time.sleep(0.4)

        return records

    all_records: Dict[str, Dict[str, object]] = {}
    max_workers = max(1, int(max_workers))

    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_country, iso): iso for iso in country_codes}
        for future in as_completed(futures):
            iso = futures[future]
            try:
                country_records = future.result()
            except Exception as exc:  # pragma: no cover - surfaced to caller
                errors.append(f"{iso}: {exc}")
                continue

            for qid_item, data in country_records.items():
                existing = all_records.get(qid_item)
                if not existing:
                    all_records[qid_item] = data
                    continue

                # Mezcla heurística para conservar información más rica
                if not existing.get("company_domain") and data.get("company_domain"):
                    existing["company_domain"] = data["company_domain"]
                if (data.get("employees") or 0) > (existing.get("employees") or 0):
                    existing["employees"] = data.get("employees")
                if (data.get("revenue") or 0) > (existing.get("revenue") or 0):
                    existing["revenue"] = data.get("revenue")
                existing["industry_labels"].update(data.get("industry_labels", set()))
                existing["tickers"].update(data.get("tickers", set()))

    if not all_records and errors:
        sequential_errors: List[str] = []
        for iso in country_codes:
            try:
                country_records = _fetch_country(iso)
            except Exception as exc:  # pragma: no cover
                sequential_errors.append(f"{iso} (fallback): {exc}")
                continue
            for qid_item, data in country_records.items():
                existing = all_records.get(qid_item)
                if not existing:
                    all_records[qid_item] = data
                    continue
                if not existing.get("company_domain") and data.get("company_domain"):
                    existing["company_domain"] = data["company_domain"]
                if (data.get("employees") or 0) > (existing.get("employees") or 0):
                    existing["employees"] = data.get("employees")
                if (data.get("revenue") or 0) > (existing.get("revenue") or 0):
                    existing["revenue"] = data.get("revenue")
                existing["industry_labels"].update(data.get("industry_labels", set()))
                existing["tickers"].update(data.get("tickers", set()))

        if not all_records:
            raise RuntimeError(
                "Fallo al consultar Wikidata en todos los países: "
                + "; ".join(errors + sequential_errors)
            )

    rows: List[Dict[str, object]] = []
    for data in all_records.values():
        name = (data.get("company_name") or "").strip()
        if not name:
            continue
        country = data.get("country", "")
        if country not in COUNTRY_QIDS:
            continue

        industries = sorted(data.get("industry_labels", []))
        domain = data.get("company_domain", "")
        tickers = sorted(data.get("tickers", []))
        employees = data.get("employees")
        revenue = data.get("revenue")

        # Requisitos mínimos: al menos dominio o ticker
        size = _size_bin(employees, revenue, tickers[0] if tickers else "")
        rows.append(
            {
                "qid": data["qid"],
                "company_name": name,
                "country": country,
                "industry_raw": "; ".join(industries),
                "description": data.get("description", ""),
                "employees": employees,
                "revenue": revenue,
                "company_domain": domain,
                "ticker": tickers[0] if tickers else "",
                "size_bin": size,
                "source": "wikidata",
                "source_rank": 1,
            }
        )
    return pd.DataFrame(rows)


def map_industries(df: pd.DataFrame, map_path: str = "config/industry_map.yml") -> pd.DataFrame:
    if df.empty:
        return df

    path = Path(map_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(default_cfg_mappings, fh, sort_keys=False, allow_unicode=False)

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
                legacy.append({"pattern": rf"(?i){re.escape(alias.strip())}", "to": slug})
        mappings = legacy

    patterns: List[Tuple[re.Pattern, str]] = []
    for entry in mappings:
        pat, target = entry.get("pattern"), entry.get("to")
        if pat and target:
            patterns.append((re.compile(pat), target))

    def _map(row: pd.Series) -> str:
        fields = [
            row.get("industry_raw", ""),
            row.get("description", ""),
            row.get("company_name", ""),
            row.get("company_domain", ""),
        ]
        raw = " ".join(p for p in fields if isinstance(p, str))
        for rx, target in patterns:
            if rx.search(raw):
                return target
        return ""

    out = df.copy()
    out["industry"] = out.apply(_map, axis=1)
    out = out[out["industry"].astype(bool)].copy()
    return out


def apply_sampling(df: pd.DataFrame, project_cfg: Dict[str, object]) -> pd.DataFrame:
    """Return a stratified sample honouring minimum employee thresholds."""

    output_cols = [
        "company_id","company_name","country","industry","size_bin",
        "company_domain","weight_stratum","ticker",
    ]

    def _stratified_sample(working: pd.DataFrame) -> pd.DataFrame:
        if working.empty:
            return pd.DataFrame(columns=output_cols)

        base = working.copy()
        if countries:
            base = base[base["country"].isin(countries)]
        if industries:
            base = base[base["industry"].isin(industries)]

        base = base[base["size_bin"].isin(["s", "m", "l"])]
        base = base.drop_duplicates(subset=["qid"]).sort_values(
            ["country", "industry", "employees"], ascending=[True, True, False]
        )

        strata: List[Tuple[str, str, pd.DataFrame]] = []
        if countries and industries:
            for c in countries:
                for i in industries:
                    sl = base[(base["country"] == c) & (base["industry"] == i)]
                    if not sl.empty:
                        strata.append((c, i, sl.reset_index(drop=True)))
        else:
            for (c, i), sl in base.groupby(["country", "industry"], sort=False):
                strata.append((c, i, sl.reset_index(drop=True)))

        if not strata:
            return pd.DataFrame(columns=output_cols)

        n_strata = len(strata)
        target_per_stratum = max(min_per_stratum, math.floor(total_target / n_strata))
        sampled_frames: List[pd.DataFrame] = []
        stratum_sizes: Dict[Tuple[str, str], int] = {}
        remainder: List[Tuple[str, str, pd.DataFrame]] = []

        collected = 0
        for c, i, sl in strata:
            take = min(len(sl), target_per_stratum)
            if take > 0:
                smp = sl.head(take).copy()
                sampled_frames.append(smp)
                stratum_sizes[(c, i)] = take
                collected += take
            else:
                stratum_sizes[(c, i)] = 0

            if len(sl) > take:
                remainder.append((c, i, sl.iloc[take:].copy()))

        remaining_needed = max(0, total_target - collected)
        if remaining_needed > 0 and remainder:
            remainder.sort(key=lambda item: len(item[2]), reverse=True)
            for c, i, extra in remainder:
                if remaining_needed <= 0:
                    break
                add = min(len(extra), remaining_needed)
                if add <= 0:
                    continue
                sampled_frames.append(extra.head(add).copy())
                stratum_sizes[(c, i)] = stratum_sizes.get((c, i), 0) + add
                remaining_needed -= add
                collected += add

        if not sampled_frames:
            return pd.DataFrame(columns=output_cols)

        sample_df = pd.concat(sampled_frames, ignore_index=True).sort_values(
            ["country", "industry", "company_name"]
        ).reset_index(drop=True)

        weights: Dict[Tuple[str, str], float] = {}
        for key, cnt in stratum_sizes.items():
            if cnt > 0:
                weights[key] = total_target / (n_strata * cnt)

        sample_df["weight_stratum"] = sample_df.apply(
            lambda r: weights.get((r["country"], r["industry"]), 1.0), axis=1
        )

        ordered_qids = sorted(sample_df["qid"].tolist())
        qid_to_id = {qid: idx + 1 for idx, qid in enumerate(ordered_qids)}
        sample_df["company_id"] = sample_df["qid"].map(qid_to_id)

        return sample_df[output_cols].sort_values("company_id").reset_index(drop=True)

    # Lee configuraciones de project.yml
    sample_cfg = (project_cfg or {}).get("sample", {}) or {}
    countries = sample_cfg.get("countries") or (project_cfg or {}).get("countries")
    industries = sample_cfg.get("industries") or (project_cfg or {}).get("industries")
    min_per_stratum = int(sample_cfg.get("min_per_stratum", (project_cfg or {}).get("min_per_stratum", 20)))
    total_target = int(sample_cfg.get("total_target", (project_cfg or {}).get("total_target", 1200)))

    raw_thresholds = sample_cfg.get("min_employees_options")
    if not raw_thresholds:
        raw_thresholds = [sample_cfg.get("min_employees", 0)]
    thresholds = [int(t) for t in raw_thresholds if t is not None]
    if not thresholds:
        thresholds = [0]

    df = df.copy()
    df["employees"] = pd.to_numeric(df.get("employees"), errors="coerce")

    def _mask_by_threshold(base: pd.DataFrame, threshold: int) -> pd.Series:
        if threshold <= 0:
            return pd.Series(True, index=base.index)

        emp = base["employees"].fillna(0)
        mask = emp >= threshold

        size_bin = base.get("size_bin")
        ticker_series = base.get("ticker")

        if threshold <= 100:
            if size_bin is not None:
                mask |= size_bin.isin(["m", "l"])
            if ticker_series is not None:
                mask |= ticker_series.fillna("").astype(str).str.len() > 0
        elif threshold <= 250:
            if size_bin is not None:
                mask |= size_bin.isin(["m", "l"])
        else:
            if size_bin is not None:
                mask |= size_bin.isin(["l"])
        return mask

    last_sample = pd.DataFrame(columns=output_cols)
    for idx, threshold in enumerate(thresholds):
        working_mask = _mask_by_threshold(df, threshold)
        working = df[working_mask].copy()

        sample_df = _stratified_sample(working)
        if not sample_df.empty:
            sample_df.attrs["min_employees_threshold"] = threshold
        last_sample = sample_df

        if len(sample_df) >= total_target:
            print(
                f"[apply_sampling] objetivo cubierto con umbral ≥{threshold} empleados (n={len(sample_df)})."
            )
            break

        if idx < len(thresholds) - 1:
            print(
                f"[apply_sampling] muestra parcial (n={len(sample_df)}) con umbral ≥{threshold}; probando siguiente umbral {thresholds[idx + 1]}"
            )
            continue

        if len(sample_df) >= total_target or idx == len(thresholds) - 1:
            break
    if last_sample.empty:
        print("[apply_sampling] advertencia: no se pudo construir muestra con los umbrales configurados.")
    elif len(last_sample) < total_target:
        print(
            f"[apply_sampling] advertencia: muestra final n={len(last_sample)} por debajo del objetivo {total_target}."
        )
    return last_sample
