# src/ctr25/signals/finance.py
from __future__ import annotations
import os, re, time, math, csv
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin
import pandas as pd
import requests, requests_cache, backoff
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

DATA_DIR="data"; RAW_DIR=os.path.join(DATA_DIR,"raw","finance")
PROC_DIR=os.path.join(DATA_DIR,"processed")
INTERIM=os.path.join(DATA_DIR,"interim"); CACHE=os.path.join(INTERIM,"cache")
EVENTS_PATH=os.path.join(PROC_DIR,"events_normalized.csv")
CACHE_PATH=os.path.join(CACHE,"finance.sqlite")
UA="ctr25-scraper-finance/1.0"
WINDOW_DAYS=365

NEWS_PATHS=["/news","/press","/media","/noticias","/prensa","/investors","/relacion-con-inversores","/relacoes-com-investidores"]

STRONG_RX = [
  re.compile(r"\bgreen bond(s)?\b", re.I),
  re.compile(r"\bsustainability[- ]linked loan(s)?\b|\bSLL\b", re.I),
]
AMOUNT_RX = re.compile(r"(\$|USD|US\$|R\$|MXN|ARS|CLP|COP|U\$S|€)\s?(\d[\d\., ]{2,})", re.I)

# asegurar estructura antes de inicializar cache
for _dir in (RAW_DIR, PROC_DIR, CACHE, os.path.join(INTERIM, "logs")):
    os.makedirs(_dir, exist_ok=True)

requests_cache.install_cache(CACHE_PATH, backend="sqlite", expire_after=86400)
session = requests.Session(); session.headers.update({"User-Agent":UA})

@backoff.on_exception(backoff.expo, (requests.RequestException,), max_time=60)
def http_get(u): 
    r=session.get(u, timeout=20)
    if not getattr(r,"from_cache",False): time.sleep(0.8)
    r.raise_for_status(); return r

def canonical(u:str)->str:
    try:
        p=urlparse(u)
        q=[(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True)
           if not (k.lower().startswith("utm_") or k.lower() in {"gclid","fbclid"})]
        return urlunparse((p.scheme,p.netloc.lower(),p.path.rstrip("/"),"",urlencode(q,doseq=True),""))
    except Exception: return u

def parse_date(soup:BeautifulSoup, url:str)->datetime|None:
    t=soup.find("time")
    if t and t.get("datetime"):
        dt=dateparser.parse(t["datetime"])
        if dt and not dt.tzinfo: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    meta=soup.find("meta", attrs={"property":"article:published_time"}) or soup.find("meta", attrs={"name":"date"})
    if meta and meta.get("content"):
        dt=dateparser.parse(meta["content"])
        if dt and not dt.tzinfo: dt=dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    m=re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", url)
    if m:
        y,mn,d=map(int,m.groups())
        return datetime(y,mn,d,tzinfo=timezone.utc)
    return None

def find_pages(domain:str)->list[str]:
    base=f"https://{domain.strip('/')}"
    return [base+p for p in NEWS_PATHS]+[base+"/"]

def density_strong(text:str)->float:
    tokens=max(120, len(re.findall(r"\w+", text or "")))
    hits=sum(bool(rx.search(text or "")) for rx in STRONG_RX)
    return min(1.0, (hits/tokens)*200)

def classify(text:str)->str|None:
    t=text.lower()
    if re.search(r"green bond", t): return "green_bond"
    if re.search(r"sustainability[- ]linked loan|sll", t): return "sll"
    return None

def score(ts:datetime|None, text:str)->float:
    rec=0.7
    if ts:
        age=max(0,(datetime.now(timezone.utc)-ts).days)
        rec=math.exp(-age/365)
    amount=1 if AMOUNT_RX.search(text or "") else 0
    return round(min(1.0, rec * (1.0+0.25*amount) * density_strong(text or "")),4)

def run_collect_finance(universe_path="data/processed/universe_sample.csv",
                        country:str|None=None, industry:str|None=None, max_companies:int=0)->int:
    os.makedirs(RAW_DIR, exist_ok=True); os.makedirs(PROC_DIR, exist_ok=True)
    df=pd.read_csv(universe_path, dtype=str)
    req={"company_id","company_name","country","industry","size_bin","company_domain"}
    miss=req - set(df.columns)
    if miss: raise ValueError(f"Universe missing columns: {sorted(miss)}")
    if country: df=df[df["country"]==country]
    if industry: df=df[df["industry"]==industry]
    if max_companies>0: df=df.head(max_companies)

    events=[]

    for _,row in df.iterrows():
        cid=str(row["company_id"]); cname=row.get("company_name"); ctry=row.get("country"); ind=row.get("industry")
        domain=(row.get("company_domain") or "").strip().lower()
        if not domain: continue
        for page in find_pages(domain):
            try:
                r=http_get(page); soup=BeautifulSoup(r.text,"html.parser")
            except Exception:
                continue
            # links salientes a notas
            links=[]
            for a in soup.find_all("a", href=True):
                href=a["href"]; full=href if href.startswith("http") else urljoin(page, href)
                if domain not in urlparse(full).netloc: 
                    continue
                links.append(canonical(full))
            links=list(dict.fromkeys(links))[:50]
            raw=[]
            for u in links:
                try:
                    rr=http_get(u); sp=BeautifulSoup(rr.text,"html.parser")
                except Exception: 
                    continue
                title = (sp.find("meta", property="og:title") or {}).get("content") if sp.find("meta", property="og:title") else (sp.title.get_text(" ", strip=True) if sp.title else "")
                body = " ".join(p.get_text(" ", strip=True) for p in sp.find_all("p")[:12])
                text = f"{title}\n{body}"
                st = classify(text)
                if not st: 
                    continue
                ts = parse_date(sp, u) or datetime.now(timezone.utc)
                if (datetime.now(timezone.utc)-ts).days>WINDOW_DAYS:
                    continue
                s = score(ts, text)
                if s<=0: 
                    continue
                # raw
                raw.append({
                  "company_id": cid, "url": u, "ts": ts.isoformat(), "title": title[:300],
                  "text_snippet": body[:500], "matched_keywords":"", "status":"ok", "source_page": page
                })
                events.append({
                  "company_id": cid, "company_name": cname, "country": ctry, "industry": ind, "size_bin": row.get("size_bin"),
                  "source":"finance", "signal_type": st, "signal_strength": s, "ts": ts.isoformat(),
                  "url": u, "title": title[:300], "text_snippet": body[:1000]
                })
            # escribir raw por empresa/estrato
            if raw:
                d=os.path.join("data","raw","finance",f"{ctry}_{ind}"); os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{cid}.csv"), "w", encoding="utf-8", newline="") as f:
                    w=csv.DictWriter(f, fieldnames=["company_id","url","ts","title","text_snippet","matched_keywords","status","source_page"])
                    w.writeheader(); [w.writerow(r) for r in raw]

    # dedupe global por (cid, type, url)
    key=lambda r:(r["company_id"], r["signal_type"], canonical(r["url"]))
    ded=[]; seen=set()
    for e in events:
        k=key(e)
        if k in seen: continue
        seen.add(k); ded.append(e)

    if ded:
        df=pd.DataFrame(ded)
        if os.path.exists(EVENTS_PATH): df.to_csv(EVENTS_PATH, mode="a", header=False, index=False, encoding="utf-8")
        else: df.to_csv(EVENTS_PATH, index=False, encoding="utf-8")
    return len(ded)
