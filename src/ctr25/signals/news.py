# src/ctr25/signals/news.py
from __future__ import annotations
import os, re, time, math, json, hashlib, logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
import csv
import pandas as pd
import requests
import requests_cache
import backoff
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from urllib import robotparser
import yaml

# --------------------------
# Config & constants
# --------------------------
DATA_DIR = os.path.join("data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "news")
PROC_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
CACHE_DIR = os.path.join(INTERIM_DIR, "cache")
LOG_DIR = os.path.join(INTERIM_DIR, "logs")
EVENTS_PATH = os.path.join(PROC_DIR, "events_normalized.csv")
CACHE_PATH = os.path.join(CACHE_DIR, "news.sqlite")

NEWSROOM_PATHS = ["/news", "/press", "/media", "/noticias", "/prensa", "/imprensa", "/sala-de-imprensa"]
MAX_PAGES = 8
WINDOW_DAYS = 365  # 12 meses
USER_AGENT = "ctr25-scraper-news/1.0 (+https://example.local/ctr25)"  # ajusta si querés

DEFAULT_KEYWORDS = {
    "strong": [
        r"\bgreen bond(s)?\b", r"\bsustainability[- ]linked loan(s)?\b", r"\bSLL\b",
        r"\bbiogas\b", r"\bhydrogen\b", r"\bH2\b", r"\bCCUS\b", r"\bcarbon capture\b",
        r"\bMRV\b", r"\bISO\s*14001\b", r"\bTCFD\b", r"\bTNFD\b", r"\bCDP\b",
        r"\bUN\s*Global\s*Compact\b|\bPacto\s*Global\b", r"\bB\s*Corp\b|\bSistema\s*B\b",
        r"\bSBTi\b", r"\bRE100\b"
    ],
    "general": [
        r"\bpilot(s)?\b|\bpiloto(s)?\b|\bprojeto piloto\b",
        r"\brenewable(s)?\b|\brenovable(s)?\b|\brenovável(eis)?\b",
        r"\bnet ?zero\b", r"\bdecarboni[sz]ation\b|\bdescarbonización\b|\bdescarbonização\b",
        r"\bESG\b", r"\bsustainab(le|ility)\b|\bsustentable(s)?\b|\bsustentabilidade\b"
    ]
}

AGGREGATOR_CFG_DEFAULT = {"enabled": False, "provider": "gnews", "api_key": None}

REQUIRED_COLS = {"company_id","company_name","country","industry","size_bin","company_domain"}

def _load_universe(universe_path: str) -> pd.DataFrame:
    if not os.path.exists(universe_path) or os.path.getsize(universe_path) == 0:
        raise FileNotFoundError(f"Universe file missing or empty: {universe_path}")
    df = pd.read_csv(universe_path, dtype=str)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Universe missing required columns: {sorted(missing)}")
    return df

# --------------------------
# Utils
# --------------------------
def ensure_dirs():
    for p in [RAW_DIR, PROC_DIR, CACHE_DIR, LOG_DIR]:
        os.makedirs(p, exist_ok=True)

def load_yaml(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    return default

def canonicalize_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        # drop tracking params
        q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not (k.lower().startswith("utm_") or k.lower() in {"gclid","fbclid"})]
        new_query = urlencode(q, doseq=True)
        # remove fragment; normalize path trailing slash
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        canon = urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", new_query, ""))
        return canon
    except Exception:
        return u

def within_window(dt: datetime, days: int = WINDOW_DAYS) -> bool:
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=days)

def parse_any_date(s: str) -> datetime | None:
    try:
        dt = dateparser.parse(s)
        if not dt:
            return None
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def robots_allows(base_url: str, path: str) -> bool:
    try:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, path)
    except Exception:
        # Si falla robots, ser conservador y permitir solo ruta raíz
        return path in ["/", ""]

def hash_key(*parts) -> str:
    return hashlib.sha256(("||".join(parts)).encode("utf-8")).hexdigest()[:16]

def tokenize(txt: str) -> list[str]:
    return re.findall(r"\w+", txt.lower(), flags=re.U)

def keyword_matchers(keywords_cfg: dict) -> tuple[list[re.Pattern], list[re.Pattern]]:
    strong = [re.compile(p, re.I) for p in keywords_cfg.get("strong", [])]
    general = [re.compile(p, re.I) for p in keywords_cfg.get("general", [])]
    return strong, general

def score_signal(text: str, ts: datetime, strong_rx: list[re.Pattern], general_rx: list[re.Pattern]) -> tuple[float, list[str]]:
    txt = text or ""
    tokens = max(200, len(tokenize(txt)))
    matches = []
    strong_hits = sum(bool(r.search(txt)) or 0 for r in strong_rx)
    general_hits = sum(bool(r.search(txt)) or 0 for r in general_rx)
    # list matched keyword labels (best-effort)
    for r in strong_rx + general_rx:
        m = r.search(txt)
        if m:
            matches.append(m.group(0)[:60])

    weighted = (2.0 * strong_hits) + (1.0 * general_hits)
    density = weighted / tokens
    age_days = max(0.0, (datetime.now(timezone.utc) - ts).days)
    recency = math.exp(-age_days / 365.0)
    raw = recency * density * 100.0  # scale up a bit
    # squash to 0..1 with soft cap
    strength = max(0.0, min(1.0, raw))
    return strength, matches

def infer_signal_type(text: str, strong_rx: list[re.Pattern]) -> str:
    t = text.lower()
    if re.search(r"green bond|sustainability[- ]linked loan|sll", t, re.I):
        return "finance_green"
    if re.search(r"\bpilot(s)?\b|\bpiloto(s)?\b|\bprojeto piloto\b", t, re.I):
        return "pilot_news"
    # fallback
    return "newsroom"

def detect_rss(soup: BeautifulSoup, base: str) -> str | None:
    for link in soup.find_all("link", attrs={"rel": "alternate"}):
        if (link.get("type") or "").lower() in {"application/rss+xml", "application/atom+xml"}:
            href = link.get("href")
            if href:
                return urljoin(base, href)
    return None

# --------------------------
# HTTP with caching & backoff
# --------------------------
requests_cache.install_cache(CACHE_PATH, backend="sqlite", expire_after=86400)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "es-AR,es;q=0.9,en;q=0.8,pt;q=0.7"})

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60)
def http_get(url: str, **kw) -> requests.Response:
    resp = session.get(url, timeout=kw.get("timeout", 15))
    # gentle rate limit
    if not getattr(resp, "from_cache", False):
        time.sleep(0.9)  # ~1 req/s
    resp.raise_for_status()
    return resp

# --------------------------
# Crawling
# --------------------------
def find_newsroom_urls(domain: str) -> list[str]:
    base = f"https://{domain.strip('/')}"
    out = []
    for path in NEWSROOM_PATHS:
        if robots_allows(base, path):
            out.append(urljoin(base, path))
    # include homepage last (por si hay “news” en portada)
    if robots_allows(base, "/"):
        out.append(base + "/")
    return out

def extract_articles_from_page(url: str) -> list[dict]:
    out = []
    resp = http_get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    # RSS?
    rss = detect_rss(soup, url)
    if rss:
        try:
            rss_resp = http_get(rss)
            rss_soup = BeautifulSoup(rss_resp.text, "xml")
            for item in rss_soup.find_all(["item", "entry"]):
                link = (item.link.get("href") if item.find("link") and item.link.get("href") else (item.link.text if item.find("link") else None))
                title = item.title.text.strip() if item.find("title") else ""
                desc = item.description.text.strip() if item.find("description") else (item.summary.text.strip() if item.find("summary") else "")
                pub = item.pubDate.text if item.find("pubDate") else (item.updated.text if item.find("updated") else (item.published.text if item.find("published") else ""))
                out.append({"url": link, "title": title, "snippet": desc, "ts_raw": pub, "source_page": url})
            return out
        except Exception:
            pass
    # HTML generic extraction
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        title = a.get_text(" ", strip=True)
        if not title or len(title) < 6:
            continue
        # heurística de noticias: busca padres tipo article/card/post
        if a.find_parent(["article", "section", "div"]):
            out.append({"url": href, "title": title, "snippet": "", "ts_raw": "", "source_page": url})
    # meta article dates
    for art in soup.find_all(["article"]):
        t = art.find(["time"])
        if t and t.get("datetime"):
            pub = t["datetime"]
            a = art.find("a", href=True)
            if a:
                href = urljoin(url, a["href"])
                title = a.get_text(" ", strip=True)
                if title:
                    out.append({"url": href, "title": title, "snippet": "", "ts_raw": pub, "source_page": url})
    # dedupe in-page
    seen = set()
    deduped = []
    for it in out:
        cu = canonicalize_url(it["url"])
        if cu not in seen:
            seen.add(cu)
            it["url"] = cu
            deduped.append(it)
    return deduped

def paginate_candidates(url: str) -> list[str]:
    urls = [url]
    # Heurísticas comunes
    patterns_query = ["?page={i}", "?p={i}", "?start={ofs}", "?offset={ofs}"]
    patterns_path = ["/page/{i}", "/pagina/{i}", "/p/{i}"]

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    # Query-based pagination
    for i in range(2, MAX_PAGES + 1):
        urls.append(base + patterns_path[0].format(i=i))
        urls.append(base + patterns_path[1].format(i=i))
        urls.append(base + patterns_path[2].format(i=i))
    # start/offset
    for ofs in range(10, 10 * MAX_PAGES + 1, 10):
        urls.append(base + patterns_query[2].format(ofs=ofs))
        urls.append(base + patterns_query[3].format(ofs=ofs))
    # simple ?page=
    for i in range(2, MAX_PAGES + 1):
        sep = "&" if "?" in url else "?"
        urls.append(url + f"{sep}page={i}")
        urls.append(url + f"{sep}p={i}")
    # dedupe
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[: (MAX_PAGES + 15)]

def fetch_article_detail(u: str) -> tuple[str, str, datetime | None]:
    try:
        r = http_get(u)
        soup = BeautifulSoup(r.text, "html.parser")
        # title
        title = soup.find("meta", property="og:title")
        title = title["content"].strip() if title and title.get("content") else (soup.title.get_text(" ", strip=True) if soup.title else "")
        # text snippet
        paras = soup.find_all(["p", "li"])
        text = " ".join(p.get_text(" ", strip=True) for p in paras[:10])[:2000]
        # time
        t = soup.find("time")
        if t and t.get("datetime"):
            ts = parse_any_date(t["datetime"])
        else:
            meta = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("meta", attrs={"name": "date"})
            ts = parse_any_date(meta["content"]) if meta and meta.get("content") else None
        # fallback: try in-URL date
        if not ts:
            m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", u)
            if m:
                year, month, day = map(int, m.groups())
                ts = datetime(year, month, day, tzinfo=timezone.utc)
        return title, text, ts
    except Exception:
        return "", "", None

def collect_news_for_company(row: dict, keywords_cfg: dict, agg_cfg: dict) -> list[dict]:
    company_id = str(row["company_id"])
    domain = (row.get("company_domain") or "").strip().lower()
    out_rows = []
    raw_rows = []

    strong_rx, general_rx = keyword_matchers(keywords_cfg)
    if not domain:
        return []

    # 1) newsroom candidates
    for base_url in find_newsroom_urls(domain):
        pages = [base_url] + paginate_candidates(base_url)
        for purl in pages:
            try:
                items = extract_articles_from_page(purl)
            except Exception:
                continue
            for it in items:
                article_url = canonicalize_url(it["url"])
                if not article_url.startswith("http"):
                    continue
                title, text, ts = fetch_article_detail(article_url)
                ts = ts or parse_any_date(it.get("ts_raw") or "") or datetime.now(timezone.utc)
                if not within_window(ts):
                    continue
                strength, matches = score_signal(f"{title}\n{text}", ts, strong_rx, general_rx)
                if strength <= 0:
                    continue

                # signal_type
                s_type = infer_signal_type(f"{title}\n{text}", strong_rx)

                raw_rows.append({
                    "company_id": company_id,
                    "url": article_url,
                    "ts": ts.isoformat(),
                    "title": title,
                    "text_snippet": text[:500],
                    "matched_keywords": ";".join(matches),
                    "status": "ok",
                    "source_page": purl
                })

                out_rows.append({
                    "company_id": company_id,
                    "company_name": row.get("company_name"),
                    "country": row.get("country"),
                    "industry": row.get("industry"),
                    "size_bin": row.get("size_bin"),
                    "source": "newsroom",
                    "signal_type": s_type,
                    "signal_strength": round(strength, 4),
                    "ts": ts.isoformat(),
                    "url": article_url,
                    "title": title[:300],
                    "text_snippet": text[:1000]
                })
    # TODO: agregador opcional (interfaz), actualmente no activo por default
    return out_rows, raw_rows

def dedupe_events(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (str(r["company_id"]), r["signal_type"], canonicalize_url(r["url"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def write_raw_csv(country: str, industry: str, company_id: str, raw_rows: list[dict]):
    path_dir = os.path.join(RAW_DIR, f"{country}_{industry}")
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, f"{company_id}.csv")
    fieldnames = ["company_id", "url", "ts", "title", "text_snippet", "matched_keywords", "status", "source_page"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

def append_events_normalized(events: list[dict]):
    os.makedirs(PROC_DIR, exist_ok=True)
    df = pd.DataFrame(events)
    if os.path.exists(EVENTS_PATH):
        df.to_csv(EVENTS_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(EVENTS_PATH, index=False, encoding="utf-8")

def log_stratum(country: str, industry: str, n_companies: int, n_urls_checked: int, n_signals: int, n_errors: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"news_{country}_{industry}.log.csv")
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts_run", "country", "industry", "n_companies", "n_urls_checked", "n_signals", "n_errors", "coverage_%"])
        coverage = round((n_signals / max(1, n_companies)) * 100.0, 1)
        w.writerow([datetime.utcnow().isoformat(), country, industry, n_companies, n_urls_checked, n_signals, n_errors, coverage])

def run_collect_news(universe_path: str = os.path.join(PROC_DIR, "universe_sample.csv"),
                     keywords_path: str = os.path.join("config", "keywords.yml"),
                     aggregator_path: str = os.path.join("config", "news.yml"),
                     country: str | None = None, industry: str | None = None, max_companies: int = 0):
    
    df = _load_universe(universe_path)
    
    ensure_dirs()
    # carga universe
    df = pd.read_csv(universe_path, dtype=str)
    if country:
        df = df[df["country"] == country]
    if industry:
        df = df[df["industry"] == industry]
    if max_companies and max_companies > 0:
        df = df.head(max_companies)

    keywords_cfg = load_yaml(keywords_path, {"news_keywords": DEFAULT_KEYWORDS})
    news_keywords = keywords_cfg.get("news_keywords", DEFAULT_KEYWORDS)
    agg_cfg = load_yaml(aggregator_path, AGGREGATOR_CFG_DEFAULT)

    n_errors = 0
    n_urls_checked_total = 0
    all_events = []

    for (c, ind), g in df.groupby(["country", "industry"], dropna=False):
        for _, row in g.iterrows():
            try:
                events, raw = collect_news_for_company(row, news_keywords, agg_cfg)
                n_urls_checked_total += len(raw)
                # dedupe por empresa dentro del batch
                events = dedupe_events(events)
                all_events.extend(events)
                # raw per-company
                write_raw_csv(row.get("country", "NA"), row.get("industry", "NA"), str(row["company_id"]), raw)
            except Exception as e:
                n_errors += 1
                continue
        # log por estrato
        log_stratum(c, ind, len(g), n_urls_checked_total, len(all_events), n_errors)

    # dedupe global y append
    all_events = dedupe_events(all_events)
    if all_events:
        append_events_normalized(all_events)
    return len(all_events)
