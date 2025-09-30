# src/ctr25/signals/webscan.py
from __future__ import annotations
import os, re, math, time, csv
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
import requests, requests_cache, backoff
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as dateparser
import yaml

DATA_DIR="data"; RAW_DIR=os.path.join(DATA_DIR,"raw","webscan")
PROC_DIR=os.path.join(DATA_DIR,"processed")
INTERIM=os.path.join(DATA_DIR,"interim"); CACHE=os.path.join(INTERIM,"cache")
EVENTS_PATH=os.path.join(PROC_DIR,"events_normalized.csv")
CACHE_PATH=os.path.join(CACHE, "webscan.sqlite")
UA="ctr25-scraper-webscan/1.0"
WINDOW_PAGES=15

DISCOVERY = ["/sustainability","/sustainability/esg","/esg","/sostenibilidad","/sustentabilidad","/impact","/impacto","/responsabilidad-social","/investors","/relacion-con-inversores","/relacoes-com-investidores"]
ANCHOR_RX = re.compile(r"(sustainab|sostenib|sustentab|responsab|impact|esg)", re.I)

DEFAULT_TERMS = [
  r"net\s?zero", r"\bESG\b", r"carbon(?!ated)|carb[oó]n(?!ico)", r"emissions?|emisiones|emiss[oõ]es",
  r"renewab|renovabl|renov[aá]vel", r"biodivers|nature|naturaleza|natureza",
  r"climate|clim[aá]tico", r"CCUS|carbon capture|captura de carbono", r"\bMRV\b", r"TCFD|TNFD|CDP|ISO\s*14001"
]

# asegurar estructura necesaria antes de inicializar cache
for _dir in (RAW_DIR, PROC_DIR, CACHE, os.path.join(INTERIM, "logs")):
    os.makedirs(_dir, exist_ok=True)

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

def parse_date(soup: BeautifulSoup, url: str) -> datetime|None:
    t=soup.find("time")
    if t and t.get("datetime"):
        dt=dateparser.parse(t["datetime"]); 
        if dt and not dt.tzinfo: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    meta = soup.find("meta", attrs={"property":"article:published_time"}) or soup.find("meta", attrs={"name":"date"})
    if meta and meta.get("content"):
        dt=dateparser.parse(meta["content"]); 
        if dt and not dt.tzinfo: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    m=re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", url)
    if m:
        y,mo,da=map(int, m.groups())
        return datetime(y,mo,da,tzinfo=timezone.utc)
    return None

def term_density(text: str, terms_rx: list[re.Pattern]) -> float:
    tokens = max(250, len(re.findall(r"\w+", text or "", flags=re.U)))
    hits = sum(bool(rx.search(text or "")) for rx in terms_rx)
    return min(0.8, (hits / tokens) * 250.0)

def score(url: str, text: str, ts: datetime|None) -> float:
    anchor_bonus = 0.2 if ANCHOR_RX.search(url) else 0.0
    if ts:
        age = max(0, (datetime.now(timezone.utc) - ts).days)
        rec = math.exp(-age/365)
    else:
        rec = 0.7
    density = term_density(text, TERM_RX)
    return round(min(1.0, rec * (anchor_bonus + density)), 4)

def load_terms(path: str) -> list[re.Pattern]:
    terms = DEFAULT_TERMS
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y=yaml.safe_load(f) or {}
            terms = y.get("webscan_terms", terms)
    return [re.compile(p, re.I) for p in terms]

TERM_RX = []  # set at runtime

def discover_pages(domain: str) -> list[str]:
    base=f"https://{domain.strip('/')}"
    return [base+p for p in DISCOVERY] + [base+"/"]

def extract_text(soup: BeautifulSoup) -> str:
    parts=[]
    for tag in soup.find_all(["h1","h2","h3","p","li"]):
        parts.append(tag.get_text(" ", strip=True))
        if len(parts) > 200: break
    return " ".join(parts)

def crawl_web_esg(domain: str) -> list[dict]:
    out=[]; visited=set()
    for d in discover_pages(domain):
        try:
            r=http_get(d); soup=BeautifulSoup(r.text, "html.parser")
        except Exception:
            continue
        # primaria
        title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:200]
        text = extract_text(soup)
        ts = parse_date(soup, d)
        s = score(d, f"{title}\n{text}", ts)
        if s>0:
            out.append({"url": canonical(d), "title": title, "text": text, "ts": ts})
        # un salto
        links=[]
        for a in soup.find_all("a", href=True):
            href=a["href"]
            full=href if href.startswith("http") else urljoin(d, href)
            if domain not in urlparse(full).netloc:
                continue
            cu=canonical(full)
            if cu in visited: continue
            visited.add(cu)
            links.append(cu)
        for u in links[:WINDOW_PAGES]:
            try:
                rr=http_get(u); sp=BeautifulSoup(rr.text, "html.parser")
            except Exception:
                continue
            title = (sp.title.get_text(" ", strip=True) if sp.title else "")[:200]
            text = extract_text(sp)
            ts = parse_date(sp, u)
            s = score(u, f"{title}\n{text}", ts)
            if s>0:
                out.append({"url": canonical(u), "title": title, "text": text, "ts": ts})
    # dedupe por url
    seen=set(); ded=[]
    for it in out:
        if it["url"] in seen: continue
        seen.add(it["url"]); ded.append(it)
    return ded

def write_raw(country:str, industry:str, cid:str, rows:list[dict]):
    d=os.path.join(RAW_DIR, f"{country}_{industry}"); os.makedirs(d, exist_ok=True)
    path=os.path.join(d, f"{cid}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["company_id","url","ts","title","text_snippet","matched_keywords","status","source_page"])
        w.writeheader()
        for r in rows:
            w.writerow({
              "company_id": cid, "url": r["url"], "ts": (r["ts"] or datetime.now(timezone.utc)).isoformat(),
              "title": r["title"][:300], "text_snippet": (r["text"] or "")[:500],
              "matched_keywords":"", "status":"ok", "source_page":"webscan"
            })

def append_events(events: list[dict]):
    os.makedirs(PROC_DIR, exist_ok=True)
    df=pd.DataFrame(events)
    if os.path.exists(EVENTS_PATH): df.to_csv(EVENTS_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else: df.to_csv(EVENTS_PATH, index=False, encoding="utf-8")

def dedupe_events(rows: list[dict]) -> list[dict]:
    seen=set(); out=[]
    for r in rows:
        key=(str(r["company_id"]), r["signal_type"], canonical(r["url"]))
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

REQUIRED = {"company_id","company_name","country","industry","size_bin","company_domain"}

def run_collect_webscan(universe_path="data/processed/universe_sample.csv", keywords_path="config/keywords.yml",
                        country:str|None=None, industry:str|None=None, max_companies:int=0) -> int:
    global TERM_RX
    TERM_RX = load_terms(keywords_path)

    if not os.path.exists(universe_path) or os.path.getsize(universe_path)==0:
        raise FileNotFoundError(f"Universe missing or empty: {universe_path}")
    df=pd.read_csv(universe_path, dtype=str)
    miss = REQUIRED - set(df.columns)
    if miss: raise ValueError(f"Universe missing columns: {sorted(miss)}")
    if country: df=df[df["country"]==country]
    if industry: df=df[df["industry"]==industry]
    if max_companies>0: df=df.head(max_companies)

    events=[]
    for _,row in df.iterrows():
        cid=str(row["company_id"]); cname=row.get("company_name"); ctry=row.get("country"); ind=row.get("industry")
        domain=(row.get("company_domain") or "").strip().lower()
        if not domain: continue
        pages=crawl_web_esg(domain)
        raw=[]
        for it in pages:
            ts = it["ts"] or datetime.now(timezone.utc)
            s = score(it["url"], f"{it['title']}\n{it['text']}", ts)
            if s<=0: continue
            raw.append({"url": it["url"], "title": it["title"], "text": it["text"], "ts": ts})
            events.append({
              "company_id": cid, "company_name": cname, "country": ctry, "industry": ind, "size_bin": row.get("size_bin"),
              "source":"webscan", "signal_type":"web_esg", "signal_strength": s, "ts": ts.isoformat(),
              "url": it["url"], "title": it["title"][:300], "text_snippet": (it["text"] or "")[:1000]
            })
        write_raw(ctry, ind, cid, raw)
    # dedupe global y append
    events = dedupe_events(events)
    if events: append_events(events)
    return len(events)
