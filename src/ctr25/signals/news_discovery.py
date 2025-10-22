"""Discovery collector for open climate news in LATAM."""
from __future__ import annotations

import datetime as dt
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from dateutil import parser as dtparser
from langdetect import detect, LangDetectException
from rapidfuzz import fuzz, process
from tqdm import tqdm

from ctr25.utils.events import append_events

try:  # optional dependency, load lazily
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:  # optional dependency, load lazily
    from newspaper import Article
except ImportError:  # pragma: no cover
    Article = None

LATAM_ISO2 = {"UY", "AR", "BR", "PE", "CL", "CO", "MX"}
CLIMATE_LANGS = {"ENGLISH", "SPANISH", "PORTUGUESE"}
BUSINESS_TERMS = (
    "empresa",
    "empresas",
    "compan*",
    "company",
    "companies",
    "corporaci*",
)
OUT_FIELDS = (
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
    "description",
    "climate_score",
    "sentiment_label",
    "sentiment_score",
)
LANG_FILTERS = (
    "sourcelang:SPANISH",
    "sourcelang:PORTUGUESE",
)
MAX_INCLUDE_TERMS = 2
MAX_EXCLUDE_TERMS = 10
CORE_EXCLUDE_TERMS = (
    "futbol",
    "malware",
    "ciberataque",
    "ciberseguridad",
    "activistas",
    "turismo",
    "protesta",
    "museo",
    "pintura",
    "business travel",
)
LATAM_TEXT_MARKERS = {
    "MX": ("méxico", "mexico", "mexicana", "mexicano", "cdmx", "nuevo león", "yucatán"),
    "AR": ("argentina", "argentino", "argentina", "buenos aires", "córdoba", "cordoba"),
    "BR": ("brasil", "brasile", "brasileiro", "brasileira", "são paulo", "sao paulo", "rio de janeiro"),
    "CL": ("chile", "chilena", "chileno", "santiago"),
    "CO": ("colombia", "colombiana", "colombiano", "bogotá", "bogota"),
    "PE": ("perú", "peru", "peruana", "peruano", "lima"),
    "UY": ("uruguay", "uruguaya", "uruguayo", "montevideo"),
}
LATAM_DOMAIN_SUFFIXES = (
    (".com.mx", "MX"),
    (".mx", "MX"),
    (".com.br", "BR"),
    (".br", "BR"),
    (".com.ar", "AR"),
    (".ar", "AR"),
    (".com.cl", "CL"),
    (".cl", "CL"),
    (".com.co", "CO"),
    (".co", "CO"),
    (".com.pe", "PE"),
    (".pe", "PE"),
    (".com.uy", "UY"),
    (".uy", "UY"),
)
MIN_KEYWORD_HITS = 1
COMPANY_STOPWORDS = {
    "de",
    "la",
    "el",
    "do",
    "da",
    "del",
    "y",
    "the",
    "of",
    "and",
    "para",
}


@dataclass
class DiscoveryParams:
    keywords_path: str
    industry_map_path: str
    out_csv: str
    since: str
    until: str
    batch_size: int = 500
    gdelt_max: int = 250
    fetch_content_ratio: float = 0.25
    append_events_csv: bool = False
    mode: str = "general"
    universe_path: str = "data/processed/universe_sample.csv"
    events_path: str = "data/processed/events_normalized.csv"
    known_max_companies: int = 200


class _NERCache:
    models: dict[str, object] = {}


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_keywords(path: str) -> tuple[list[str], list[str]]:
    data = _load_yaml(path)
    include = [str(s).strip() for s in data.get("include", []) if str(s).strip()]
    exclude = [str(s).strip() for s in data.get("exclude", []) if str(s).strip()]
    if not include:
        raise ValueError("keywords.yml: lista 'include' vacía")
    return include, exclude


def _load_industry_map(path: str) -> dict:
    try:
        data = _load_yaml(path)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}


def _quoted(term: str) -> str:
    if re.search(r"\s|\"", term):
        return f'"{term}"'
    return term


def _build_boolean_block(terms: Iterable[str]) -> str:
    return " OR ".join(_quoted(t) for t in terms)


def _select_excludes(exclude_terms: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for term in CORE_EXCLUDE_TERMS:
        if term in exclude_terms and term not in seen:
            ordered.append(term)
            seen.add(term)
    for term in exclude_terms:
        if term not in seen:
            ordered.append(term)
            seen.add(term)
    return ordered[:MAX_EXCLUDE_TERMS]


def _compose_query(
    include_terms: Sequence[str],
    exclude_terms: Sequence[str],
    lang_filter: str,
) -> str:
    include_block = f"({_build_boolean_block(include_terms)})"
    business_block = f"({_build_boolean_block(BUSINESS_TERMS)})"
    effective_excludes = _select_excludes(exclude_terms)
    exclude_part = " ".join(f"-{_quoted(term)}" for term in effective_excludes)
    parts = [include_block, business_block, lang_filter]
    if exclude_part:
        parts.append(exclude_part)
    return " ".join(filter(None, parts))


def _infer_country_from_text(title: str, snippet: str, body: str) -> str:
    text = " ".join(filter(None, (title, snippet, body))).lower()
    for iso, markers in LATAM_TEXT_MARKERS.items():
        for marker in markers:
            if marker in text:
                return iso
    return ""


def _infer_country_from_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    if not netloc:
        return ""
    for suffix, iso in LATAM_DOMAIN_SUFFIXES:
        if netloc.endswith(suffix):
            return iso
    return ""


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed.strip().lower()


def _company_variants(name: str) -> list[str]:
    base = name.strip()
    variants = [base]
    cleaned = re.sub(
        r"\b(s\.a\.a\.|s\.a\.|sa|ltda|inc|corp|corporación|corporacao|company|compañía|companhia)\b",
        "",
        base,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned not in variants:
        variants.append(cleaned)
    core_tokens = [
        token
        for token in cleaned.split()
        if _normalize_text(token) not in COMPANY_STOPWORDS
    ]
    if core_tokens:
        core = " ".join(core_tokens)
        if core and core not in variants:
            variants.append(core)
    return variants


def _company_in_text(company: str, *segments: str) -> bool:
    if not company:
        return False
    blob = _normalize_text(" ".join(filter(None, segments)))
    if not blob:
        return False
    for variant in _company_variants(company):
        token = _normalize_text(variant)
        if token and token in blob:
            return True
    return False


def _load_known_companies(
    universe_path: str,
    events_path: str,
    max_companies: int,
) -> list[dict]:
    registry: dict[str, dict] = {}

    def _register(record: dict) -> None:
        name = record.get("company_name")
        if not isinstance(name, str) or not name.strip():
            return
        key = _normalize_text(name)
        if not key:
            return
        if key in registry:
            return
        country = (record.get("country") or "").upper()
        if country and country not in LATAM_ISO2:
            return
        registry[key] = record

    path = Path(universe_path)
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            record = {
                "company_id": row.get("company_id", ""),
                "company_qid": row.get("company_qid", row.get("qid", "")),
                "company_name": row.get("company_name", ""),
                "country": row.get("country", ""),
                "industry": row.get("industry", ""),
                "size_bin": row.get("size_bin", ""),
            }
            _register(record)

    path = Path(events_path)
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            record = {
                "company_id": row.get("company_id", ""),
                "company_qid": row.get("company_qid", ""),
                "company_name": row.get("company_name", ""),
                "country": row.get("country", ""),
                "industry": row.get("industry", ""),
                "size_bin": row.get("size_bin", ""),
            }
            _register(record)

    companies = list(registry.values())
    if max_companies and max_companies > 0:
        companies = companies[:max_companies]
    return companies


def _build_known_registry(
    companies: Sequence[dict],
    industry_map: dict,
) -> dict[str, dict]:
    registry: dict[str, dict] = {}

    def _store(record: dict, alias: str) -> None:
        key = _normalize_text(alias)
        if not key:
            return
        registry[key] = record

    for record in companies:
        name = record.get("company_name", "")
        for variant in _company_variants(name):
            _store(record, variant)
        mapping = industry_map.get(name)
        if isinstance(mapping, dict):
            for alias in (mapping.get("aliases") or []):
                for variant in _company_variants(str(alias)):
                    _store(record, variant)

    return registry


def _build_general_queries(
    include_terms: Sequence[str],
    exclude_terms: Sequence[str],
) -> list[str]:
    queries: list[str] = []
    current: list[str] = []

    def _emit(chunk: list[str]) -> None:
        if not chunk:
            return
        for lang in LANG_FILTERS:
            queries.append(_compose_query(chunk, exclude_terms, lang))

    for term in include_terms:
        current.append(term)
        if len(current) >= MAX_INCLUDE_TERMS:
            _emit(current)
            current = []
    _emit(current)

    if not queries:
        _emit(list(include_terms))
    return queries


def _request_gdelt(query: str, start_dt: dt.datetime, end_dt: dt.datetime, *,
                   maxrecords: int, tries: int = 3, pause: float = 1.0) -> list[dict]:
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(maxrecords),
        "format": "JSON",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    last_err: Exception | None = None
    for _ in range(tries):
        try:
            response = requests.get(base, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get("articles", [])
            last_err = RuntimeError(f"gdelt {response.status_code} {response.text[:160]}")
        except Exception as exc:  # pragma: no cover
            last_err = exc
        time.sleep(pause)
    if last_err:
        raise last_err
    return []

import os
os.environ.setdefault("MEDIACLOUD_API_KEY", "0424475b44b1e7dd3f7a71435e3ed3a122989f1e")

def _request_mediacloud(include_terms: Sequence[str], *, since: str, until: str, rows: int = 1000) -> list[dict]:
    key = os.getenv("MEDIACLOUD_API_KEY")
    if not key:
        return []
    query = f"({_build_boolean_block(include_terms)}) AND ({_build_boolean_block(BUSINESS_TERMS)})"
    url = (
        "https://api.mediacloud.org/api/v2/stories"
        f"?q={requests.utils.quote(query)}"
        f"&publish_date_start={since}"
        f"&publish_date_end={until}"
        f"&rows={rows}"
    )
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception:  # pragma: no cover
        return []
    out: list[dict] = []
    for story in data.get("results", []):
        out.append(
            {
                "url": story.get("url"),
                "title": story.get("title"),
                "language": story.get("language", ""),
                "sourcecountry": "",
                "seendate": story.get("publish_date", ""),
                "source": "mediacloud",
            }
        )
    return out


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "es"


def _load_model(name: str):
    if not spacy:
        return None
    cached = _NERCache.models.get(name)
    if cached:
        return cached
    try:
        model = spacy.load(name)
    except Exception:  # pragma: no cover
        model = None
    _NERCache.models[name] = model
    return model


def _pick_model(sample_text: str):
    lang = _detect_language(sample_text or "es")
    if lang.startswith("es"):
        return _load_model("es_core_news_lg") or _load_model("es_core_news_md")
    if lang.startswith("pt"):
        return _load_model("pt_core_news_lg") or _load_model("pt_core_news_md")
    return _load_model("en_core_web_lg") or _load_model("en_core_web_md") or _load_model("en_core_web_sm")


def _extract_orgs(title: str, snippet: str, body: str) -> set[str]:
    text = " ".join(filter(None, (title, snippet, body)))
    model = _pick_model(text)
    if not model:
        return set()
    doc = model(text)
    banned = {
        "gobierno",
        "ministerio",
        "naciones unidas",
        "onu",
        "ong",
        "municipalidad",
        "universidad",
        "capítulo clima",
        "chapter clima",
        "capitulo clima",
        "business travel",
    }
    orgs: set[str] = set()
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        candidate = ent.text.strip()
        if len(candidate) < 3:
            continue
        lower = candidate.lower()
        if lower in banned:
            continue
        if ":" in candidate:
            continue
        if not any(ch.isupper() for ch in candidate if ch.isalpha()):
            continue
        orgs.add(candidate)
    return orgs


def _fetch_snippet(url: str, *, max_chars: int = 420) -> tuple[str, str]:
    if not Article:
        return "", ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
    except Exception:  # pragma: no cover
        return "", ""
    if not text:
        return "", ""
    snippet = re.sub(r"\s+", " ", text)[:max_chars].strip()
    return snippet, text


def _score_keywords(text: str, include_terms: Sequence[str]) -> tuple[float, int]:
    norm = text.lower()
    hits = sum(1 for term in include_terms if term.lower() in norm)
    denom = max(1, len({term.lower() for term in include_terms}))
    return round(hits / denom, 3), hits


def _signal_strength(source_country: str, language: str, hits: int) -> float:
    strength = 0.5
    if language.upper() in CLIMATE_LANGS:
        strength += 0.2
    if source_country.upper() in LATAM_ISO2:
        strength += 0.2
    strength += min(0.1, 0.02 * hits)
    return round(min(strength, 1.0), 3)


def _infer_country(language_guess: str) -> str:
    mapping = {"pt": "BR", "es": "", "en": ""}
    lang = language_guess[:2].lower()
    candidate = mapping.get(lang, "")
    return candidate if candidate in LATAM_ISO2 else ""


def _map_industry(name: str, industry_map: dict) -> str:
    if not name or not industry_map:
        return "unknown"
    direct = industry_map.get(name)
    if isinstance(direct, dict) and direct.get("industry"):
        return direct["industry"]
    for entry_name, payload in industry_map.items():
        aliases = []
        if isinstance(payload, dict):
            aliases = payload.get("aliases", []) or []
            if name in aliases and payload.get("industry"):
                return payload["industry"]
    choices = list(industry_map.keys())
    if not choices:
        return "unknown"
    best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if not best:
        return "unknown"
    match, score, _ = best
    if score >= 92:
        payload = industry_map.get(match)
        if isinstance(payload, dict) and payload.get("industry"):
            return payload["industry"]
    return "unknown"


def _normalize_row(company_name: str, country: str, industry: str, *, source: str, title: str,
                   url: str, ts_iso: str, text_snippet: str, description: str, climate_score: float,
                   hits: int, language: str, source_country: str) -> dict:
    return {
        "company_id": "",
        "company_qid": "",
        "company_name": company_name,
        "country": country,
        "industry": industry or "unknown",
        "size_bin": "",
        "source": source,
        "signal_type": "news",
        "signal_strength": _signal_strength(source_country, language, hits),
        "ts": ts_iso,
        "url": url,
        "title": title,
        "text_snippet": text_snippet,
        "description": description,
        "climate_score": climate_score,
        "sentiment_label": "",
        "sentiment_score": "",
    }


def _prepare_articles(gdelt_articles: list[dict], mc_articles: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for article in (*gdelt_articles, *mc_articles):
        url = article.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(article)
    return out


def run_discovery(params: DiscoveryParams) -> tuple[pd.DataFrame, int]:
    include_terms, exclude_terms = _load_keywords(params.keywords_path)
    industry_map = _load_industry_map(params.industry_map_path)
    since_dt = dtparser.parse(params.since).replace(tzinfo=dt.timezone.utc)
    until_dt = dtparser.parse(params.until).replace(tzinfo=dt.timezone.utc)

    known_registry: dict[str, dict] | None = None
    known_source_label = "news_discovery"
    if params.mode == "known":
        known_companies = _load_known_companies(
            params.universe_path,
            params.events_path,
            params.known_max_companies,
        )
        if not known_companies:
            print("[news-discovery] sin compañías conocidas para consultar")
            return pd.DataFrame(columns=OUT_FIELDS), 0
        known_registry = _build_known_registry(known_companies, industry_map)
        known_source_label = "news_discovery_known"

    if len(exclude_terms) > MAX_EXCLUDE_TERMS:
        print(
            f"[news-discovery] truncando excludes de {len(exclude_terms)} a {MAX_EXCLUDE_TERMS} para GDELT"
        )

    queries = _build_general_queries(include_terms, exclude_terms)

    gdelt_articles: list[dict] = []
    remaining = params.gdelt_max if params.gdelt_max > 0 else 250
    for idx, query in enumerate(queries):
        if remaining <= 0:
            break
        slots = len(queries) - idx
        per_query_max = max(1, remaining // max(1, slots))
        per_query_max = max(25, per_query_max)
        per_query_max = min(per_query_max, remaining)
        try:
            articles = _request_gdelt(
                query,
                since_dt,
                until_dt,
                maxrecords=per_query_max,
            )
            gdelt_articles.extend(articles)
        except Exception as exc:  # pragma: no cover
            print(f"[news-discovery] fallo GDELT chunk {idx+1}/{len(queries)}: {exc}")
        remaining -= per_query_max

    mc_articles = _request_mediacloud(include_terms, since=params.since, until=params.until, rows=params.batch_size)

    articles = _prepare_articles(gdelt_articles, mc_articles)
    rows: list[dict] = []

    to_fetch = max(0, int(len(articles) * params.fetch_content_ratio))

    for article in tqdm(articles, desc="discover-news", unit="art"):
        url = article.get("url", "")
        if not url:
            continue
        title = (article.get("title") or "").strip()
        language = (article.get("language") or "").strip()
        source_country = (article.get("sourcecountry") or "").strip()
        seendate = article.get("seendate") or ""

        if source_country and source_country.upper() not in LATAM_ISO2:
            source_country = ""

        domain_country = _infer_country_from_domain(url)
        if not source_country and domain_country:
            source_country = domain_country

        ts_iso = ""
        if seendate:
            try:
                ts = dtparser.parse(seendate)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                ts_iso = ts.astimezone(dt.timezone.utc).isoformat()
            except Exception:
                ts_iso = ""

        snippet = ""
        body = ""
        if to_fetch > 0:
            snippet, body = _fetch_snippet(url)
            to_fetch -= 1

        text_for_score = " ".join(filter(None, (title, snippet, body[:500])))
        clim_score, hits = _score_keywords(text_for_score, include_terms)
        if hits < MIN_KEYWORD_HITS:
            continue

        orgs = _extract_orgs(title, snippet, body)
        if not orgs:
            continue

        base_country = source_country if source_country in LATAM_ISO2 else ""

        for org in orgs:
            org = org.strip()
            if len(org) < 3:
                continue

            matched_record: dict | None = None
            canonical_name = org
            if known_registry is not None:
                for variant in _company_variants(org):
                    key = _normalize_text(variant)
                    if key in known_registry:
                        matched_record = known_registry[key]
                        break
                if not matched_record:
                    continue
                canonical_name = matched_record.get("company_name", org)

            industry = (
                matched_record.get("industry")
                if matched_record and matched_record.get("industry")
                else _map_industry(canonical_name, industry_map)
            )

            candidate_country = base_country
            if not candidate_country and matched_record and matched_record.get("country"):
                candidate_country = str(matched_record.get("country")).upper()
            if not candidate_country:
                inferred = _infer_country_from_text(title, snippet, body)
                if inferred:
                    candidate_country = inferred
            if not candidate_country:
                lang_guess = _detect_language(" ".join(filter(None, (title, snippet, body))))
                candidate_country = _infer_country(lang_guess)
            if not candidate_country:
                continue

            source_label = known_source_label if known_registry else (article.get("source", "gdelt") or "gdelt")
            row = _normalize_row(
                company_name=canonical_name,
                country=candidate_country,
                industry=industry,
                source=source_label,
                title=title,
                url=url,
                ts_iso=ts_iso,
                text_snippet=snippet,
                description="",
                climate_score=clim_score,
                hits=hits,
                language=language,
                source_country=source_country or candidate_country,
            )

            if matched_record:
                row["company_id"] = matched_record.get("company_id", "")
                row["company_qid"] = matched_record.get("company_qid", "")
                row["industry"] = matched_record.get("industry") or industry or "unknown"
                row["size_bin"] = matched_record.get("size_bin", "")

            if not row["company_name"]:
                continue
            rows.append(row)

    df = pd.DataFrame(rows, columns=OUT_FIELDS)

    out_path = Path(params.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    if not df.empty:
        df.to_csv(out_path, mode="a" if out_path.exists() else "w", header=write_header, index=False)

    appended = 0
    if params.append_events_csv and not df.empty:
        to_append = df[[
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
        ]].copy()
        if known_registry is not None:
            to_append["source"] = known_source_label
            appended = append_events(to_append, source=known_source_label, signal_type="news")
        else:
            to_append["source"] = "news_discovery"
            appended = append_events(to_append, source="news_discovery", signal_type="news")

    return df, appended


def collect_news_discovery(**kwargs) -> tuple[pd.DataFrame, int]:
    params = DiscoveryParams(**kwargs)
    return run_discovery(params)


def run_collect_news_discovery(**kwargs) -> tuple[pd.DataFrame, int]:
    return collect_news_discovery(**kwargs)
