# src/ctr25/signals/jobs.py
from __future__ import annotations
import os, re, time, math, csv
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin
import pandas as pd
import requests, requests_cache, backoff
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import yaml

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw", "jobs")
PROC_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
CACHE_DIR = os.path.join(INTERIM_DIR, "cache")
LOG_DIR = os.path.join(INTERIM_DIR, "logs")
EVENTS_PATH = os.path.join(PROC_DIR, "events_normalized.csv")
CACHE_PATH = os.path.join(CACHE_DIR, "jobs.sqlite")

WINDOW_DAYS = 365
UA = "ctr25-scraper-jobs/1.0"

DEFAULT_JOBS_KEYWORDS = [
  r"sustainab(le|ility)|sustentab(le|ilidad|ilidade)",
  r"\bESG\b|\bASG\b",
  r"carbon|carb[oó]n|carv[aã]o|emissions?|emisiones|emiss[oõ]es",
  r"biodiversit(y|ad|ade)|nature|naturaleza|natureza",
  r"renewable|renovable(s)?|renov[aá]vel(eis)?|energia limpia"
]

SITES = [
  ("indeed",   "https://{tld}/jobs?q={q}",       ["indeed.com", "indeed.com.br", "indeed.com.mx", "indeed.cl", "indeed.com.ar", "co.indeed.com"]),
  ("glassdoor","https://www.glassdoor.com/Job/jobs.htm?sc.keyword={q}", ["glassdoor.com"]),
  ("linkedin", "https://www.linkedin.com/jobs/search/?keywords={q}", ["linkedin.com"])
]

requests_cache.install_cache(CACHE_PATH, backend="sqlite", expire_after=86400)
session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language":"es-AR,es;q=0.9,pt;q=0.8,en;q=0.7"})

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60)
def http_get(u: str):
    r = session.get(u, timeout=20)
    if not getattr(r, "from_cache", False):
        time.sleep(0.8)
    r.raise_for_status()
    return r

def canonical(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True)
             if not (k.lower().startswith("utm_") or k.lower() in {"gclid","fbclid"})]
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", urlencode(q, doseq=True), ""))
    except Exception:
        return u

def parse_date_guess(txt: str) -> datetime|None:
    # soporta “hace N días”, “N days ago”, “N dias atrás”, fechas exactas
    t = txt.lower()
    m = re.search(r"(\d+)\s+(day|days|d[ií]as)\s+ago|hace\s+(\d+)\s+d[ií]as|(\d+)\s+dias\s+atr[aá]s", t)
    try:
        if m:
            n = int([g for g in m.groups() if g and g.isdigit()][0])
            return (datetime.now(timezone.utc) - timedelta(days=n))
        dt = dateparser.parse(txt)
        if dt:
            if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return None

SENIORITY_WEIGHTS = [
  (re.compile(r"\b(head|director|chief|lead|manager|gerente|diretor|lider)\b", re.I), 1.3),
  (re.compile(r"\b(senior|sr\.?)\b", re.I), 1.15),
  (re.compile(r"\b(junior|jr\.?)\b", re.I), 0.6),
]

def seniority_weight(title: str) -> float:
    for rx, w in SENIORITY_WEIGHTS:
        if rx.search(title or ""): return w
    return 1.0

def density_score(title: str, snippet: str, patterns: list[re.Pattern]) -> float:
    txt = " ".join([title or "", snippet or ""]).lower()
    tokens = max(120, len(re.findall(r"\w+", txt)))
    hits = sum(bool(rx.search(txt)) for rx in patterns)
    return min(1.0, (hits / tokens) * 180)

def score(ts: datetime|None, title: str, snippet: str, kw_rx: list[re.Pattern]) -> float:
    rec = 0.7
    if ts:
        age = max(0, (datetime.now(timezone.utc) - ts).days)
        rec = math.exp(-age/365)
    return round(min(1.0, rec * seniority_weight(title) * density_score(title, snippet, kw_rx)), 4)

def load_keywords(path: str) -> list[re.Pattern]:
    kw = DEFAULT_JOBS_KEYWORDS
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            kw = y.get("jobs_keywords", kw)
    return [re.compile(p, re.I) for p in kw]

def site_for_country(country: str, site: str) -> str:
    tld = "www.indeed.com"
    if site=="indeed":
        tld = {"AR":"ar.indeed.com","BR":"indeed.com.br","CL":"cl.indeed.com","MX":"mx.indeed.com","CO":"co.indeed.com","UY":"uy.indeed.com"}.get(country,"www.indeed.com")
    return tld

def fetch_listings(company_name: str, country: str, kw_query: str) -> list[dict]:
    out=[]
    for name, url_tpl, domains in SITES:
        if name=="indeed":
            tld = site_for_country(country, name)
            url = url_tpl.format(tld=tld, q=kw_query)
        else:
            url = url_tpl.format(q=kw_query)
        try:
            r = http_get(url)
            soup = BeautifulSoup(r.text, "html.parser")
            # heurísticas generales para tarjetas
            for a in soup.find_all("a", href=True):
                href = a["href"]
                title = a.get_text(" ", strip=True)
                if not title or len(title) < 6: 
                    continue
                full = href if href.startswith("http") else urljoin(url, href)
                # filtros: dominio objetivo
                if not any(d in urlparse(full).netloc for d in domains):
                    continue
                # contextos cercanos
                parent = a.find_parent(["article","div","li"])
                snippet = ""
                date_txt = ""
                if parent:
                    ptxt = parent.get_text(" ", strip=True)
                    snippet = ptxt[:280]
                    # fechas típicas
                    mt = re.search(r"(hace\s+\d+\s+d[ií]as|\d+\s+(days?|d[ií]as)\s+ago|\b\d{4}-\d{2}-\d{2}\b)", ptxt, re.I)
                    date_txt = mt.group(0) if mt else ""
                out.append({"url": canonical(full), "title": title, "snippet": snippet, "date_txt": date_txt, "source_page": url})
        except Exception:
            continue
    # dedupe por url
    seen=set(); ded=[]
    for it in out:
        u=it["url"]
        if u not in seen:
            seen.add(u); ded.append(it)
    return ded[:60]

def write_raw(country:str, industry:str, cid:str, rows:list[dict]):
    d = os.path.join(RAW_DIR, f"{country}_{industry}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{cid}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["company_id","url","ts","title","text_snippet","matched_keywords","status","source_page"])
        w.writeheader()
        for r in rows: w.writerow(r)

def append_events(events: list[dict]):
    os.makedirs(PROC_DIR, exist_ok=True)
    df=pd.DataFrame(events)
    if os.path.exists(EVENTS_PATH):
        df.to_csv(EVENTS_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(EVENTS_PATH, index=False, encoding="utf-8")

def dedupe_events(rows: list[dict]) -> list[dict]:
    seen=set(); out=[]
    for r in rows:
        key=(str(r["company_id"]), r["signal_type"], canonical(r["url"]))
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

REQUIRED = {"company_id","company_name","country","industry","size_bin","company_domain"}

def run_collect_jobs(universe_path="data/processed/universe_sample.csv", keywords_path="config/keywords.yml",
                     country: str|None=None, industry: str|None=None, max_companies:int=0) -> int:
    os.makedirs(RAW_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(universe_path) or os.path.getsize(universe_path)==0:
        raise FileNotFoundError(f"Universe missing or empty: {universe_path}")
    df = pd.read_csv(universe_path, dtype=str)
    miss = REQUIRED - set(df.columns)
    if miss: raise ValueError(f"Universe missing columns: {sorted(miss)}")
    if country: df = df[df["country"]==country]
    if industry: df = df[df["industry"]==industry]
    if max_companies>0: df=df.head(max_companies)

    kw_rx = load_keywords(keywords_path)
    all_events=[]; total_raw=0

    for _, row in df.iterrows():
        cid=str(row["company_id"])
        cname=(row.get("company_name") or "").strip()
        ctry=row.get("country") or "NA"
        ind=row.get("industry") or "NA"
        domain=(row.get("company_domain") or "").strip()
        if not cname:
            continue
        # query por empresa + keywords (frugal)
        kw_query = requests.utils.quote(f'"{cname}" (sustainability OR ESG OR carbono OR renovable OR natureza)')
        listings = fetch_listings(cname, ctry, kw_query)
        raw_rows=[]
        ev=[]
        for it in listings:
            ts = parse_date_guess(it.get("date_txt") or "") or datetime.now(timezone.utc)
            s = score(ts, it["title"], it["snippet"], kw_rx)
            if s<=0: continue
            raw_rows.append({
              "company_id": cid, "url": it["url"], "ts": ts.isoformat(), "title": it["title"][:300],
              "text_snippet": (it["snippet"] or "")[:500], "matched_keywords":"", "status":"ok", "source_page": it["source_page"]
            })
            ev.append({
              "company_id": cid, "company_name": cname, "country": ctry, "industry": ind, "size_bin": row.get("size_bin"),
              "source":"jobs", "signal_type":"job_posting", "signal_strength": s, "ts": ts.isoformat(),
              "url": it["url"], "title": it["title"][:300], "text_snippet": (it["snippet"] or "")[:1000]
            })
        total_raw += len(raw_rows)
        write_raw(ctry, ind, cid, raw_rows)
        all_events.extend(dedupe_events(ev))

    if all_events: append_events(all_events)
    return len(all_events)
