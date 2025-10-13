import csv
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://unglobalcompact.org"
SEARCH_PATH = "/what-is-gc/participants/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EA-CTR25-scraper/1.0; +https://emergenteanalytics.la)"
}
TARGET_ISO_TO_LABEL = {
    "AR": "Argentina",
    "BR": "Brazil",
    "CL": "Chile",
    "CO": "Colombia",
    "MX": "Mexico",
    "PE": "Peru",
    "UY": "Uruguay",
}
PER_PAGE = 50
RATE_SLEEP = float(os.environ.get("UNGC_RATE_SLEEP", "0.8"))

def get_soup(url, params=None, session=None, retries=3):
    session = session or requests.Session()
    for attempt in range(retries):
        response = session.get(url, params=params, headers=HEADERS, timeout=30)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        time.sleep(1.5 * (attempt + 1))
    response.raise_for_status()

def discover_country_id_map(session=None):
    """Lee el formulario y arma {label -> id_value} del filtro Country."""
    soup = get_soup(urljoin(BASE, SEARCH_PATH), session=session)

    select = soup.find("select", attrs={"name": "search[countries][]"})
    if select:
        mapping = {}
        for option in select.find_all("option"):
            label = (option.get_text() or "").strip()
            value = (option.get("value") or "").strip()
            if label and value:
                mapping[label] = value
        if mapping:
            return mapping

    mapping = {}
    inputs = soup.find_all("input", attrs={"name": "search[countries][]"})
    for input_tag in inputs:
        value = (input_tag.get("value") or "").strip()
        if not value:
            continue
        label_text = ""
        input_id = input_tag.get("id")
        if input_id:
            label_tag = soup.find("label", attrs={"for": input_id})
            if label_tag:
                label_text = label_tag.get_text(strip=True)
        if not label_text:
            parent_label = input_tag.find_parent("label")
            if parent_label:
                label_text = parent_label.get_text(strip=True)
        if label_text:
            mapping[label_text] = value
    if mapping:
        return mapping

    raise RuntimeError("No se pudo localizar el selector de países en el formulario.")

def parse_table_rows(soup):
    """Devuelve lista de dicts con las columnas visibles en la tabla de resultados."""
    table = soup.find("table")
    if not table:
        return []

    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    else:
        first_row = table.find("tr")
        headers = [cell.get_text(strip=True) for cell in first_row.find_all(["th", "td"])] if first_row else []

    tbody = table.find("tbody") or table
    rows_out = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if len(values) < len(headers):
            values.extend(["" for _ in range(len(headers) - len(values))])
        data = dict(zip(headers, values[:len(headers)]))

        link = ""
        name_cell = tr.find("th", scope="row")
        if name_cell:
            anchor = name_cell.find("a", href=True)
            if anchor:
                link = anchor["href"]
        if not link:
            anchor = tr.find("a", href=True)
            if anchor:
                link = anchor["href"]

        rows_out.append({
            "name": data.get("Name", ""),
            "type": data.get("Type", ""),
            "sector": data.get("Sector", ""),
            "country": data.get("Country", ""),
            "joined_on": data.get("Joined On", ""),
            "participant_url": urljoin(BASE, link) if link else "",
        })
    return rows_out

def parse_participant_detail(url, session=None):
    """Extrae info adicional de la ficha: status, website, ownership, next_cop_due."""
    if not url:
        return {}
    soup = get_soup(url, session=session)
    info = {}

    website = ""
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("http") and "unglobalcompact.org" not in href:
            website = href
            break
    info["website"] = website

    def scrape_label(label_text):
        element = soup.find(string=lambda value: isinstance(value, str) and value.strip().lower() == label_text.lower())
        if element and element.parent:
            sibling = element.parent.find_next_sibling()
            if sibling:
                return sibling.get_text(" ", strip=True)
            return element.parent.get_text(" ", strip=True).replace(label_text, "").strip(": ").strip()
        return ""

    for label_text, key in [
        ("Global Compact Status", "status"),
        ("Ownership", "ownership"),
        ("Next Communication on Progress (COP) due on:", "next_cop_due"),
        ("Next Communication on Progress (CoP) due on:", "next_cop_due"),
    ]:
        value = scrape_label(label_text)
        if not value and "status" in key:
            value = scrape_label("Status")
        info[key] = value

    return info

def load_existing(out_csv):
    existing = {}
    path = Path(out_csv)
    if not path.exists() or path.stat().st_size == 0:
        return existing
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("participant_url") or row.get("name")
            if key:
                existing[key] = row
    return existing

def scrape_country(label, country_id, session, writer, handle, fieldnames, existing_urls):
    page = 1
    added = 0
    while True:
        params = [
            ("search[countries][]", country_id),
            ("search[per_page]", str(PER_PAGE)),
            ("page", str(page)),
            ("search[sort_field]", ""),
            ("search[sort_direction]", "asc"),
        ]
        soup = get_soup(urljoin(BASE, SEARCH_PATH), params=params, session=session)
        rows = parse_table_rows(soup)
        if not rows:
            break
        for row in rows:
            url = row.get("participant_url", "")
            if url and url in existing_urls:
                continue
            try:
                detail = parse_participant_detail(url, session=session)
            except Exception:
                detail = {}
            row.update(detail)
            payload = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(payload)
            handle.flush()
            if url:
                existing_urls.add(url)
            added += 1
            if RATE_SLEEP:
                time.sleep(RATE_SLEEP)
        page += 1
        if RATE_SLEEP:
            time.sleep(RATE_SLEEP)
    if added:
        print(f"{label}: +{added} organizaciones nuevas")
    else:
        print(f"{label}: sin novedades")
    return added

def main(out_csv="ungc_participants_latam.csv"):
    session = requests.Session()
    country_map = discover_country_id_map(session=session)
    wanted_labels = [TARGET_ISO_TO_LABEL[code] for code in TARGET_ISO_TO_LABEL]
    missing = [label for label in wanted_labels if label not in country_map]
    if missing:
        raise RuntimeError(f"No se encontraron en el selector: {missing}")

    ordered_countries = [(label, country_map[label]) for label in wanted_labels]

    fieldnames = [
        "name",
        "type",
        "sector",
        "country",
        "joined_on",
        "status",
        "ownership",
        "next_cop_due",
        "participant_url",
        "website",
    ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = load_existing(out_path)
    existing_urls = set(existing_rows)

    mode = "a" if out_path.exists() and out_path.stat().st_size > 0 else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        total_new = 0
        for label, cid in ordered_countries:
            total_new += scrape_country(label, cid, session, writer, handle, fieldnames, existing_urls)

    total_records = len(existing_urls)
    print(f"OK -> {out_csv} (total {total_records} organizaciones, +{total_new} nuevas en esta corrida)")

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "ungc_participants_latam.csv"
    main(output_path)
