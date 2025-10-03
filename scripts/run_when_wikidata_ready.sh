#!/usr/bin/env bash
set -euo pipefail

echo "[watch] esperando a que query.wikidata.org responda..."

while true; do
  if python3 - <<'PY'
import requests
r = requests.get(
  "https://query.wikidata.org/sparql",
  params={"format": "json", "query": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"},
  headers={"User-Agent": "ctr25/network-watch"},
  timeout=20,
)
if r.status_code == 200:
  print("READY")
  raise SystemExit(0)
raise SystemExit(1)
PY
  then
  echo "[watch] listo → corriendo muestreo…"
  ctr25 sample --force --per-country 120 --max-workers 4 --save-raw data/interim/qa/raw_wikidata.csv
  break
  else
  echo "[watch] todavía no responde, reintento en 60s"
  sleep 60
  fi
done
