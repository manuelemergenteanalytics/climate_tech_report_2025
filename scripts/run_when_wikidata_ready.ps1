$ErrorActionPreference = "SilentlyContinue"
Write-Host "[watch] esperando a que query.wikidata.org responda..."

$testUrl = "https://query.wikidata.org/sparql?format=json&query=SELECT%20*%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20LIMIT%201"

while ($true) {
  try {
    $resp = Invoke-WebRequest -Uri $testUrl -Method GET
    if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400) {
      Write-Host "[watch] listo -> corriendo muestreo..."
      ctr25 sample --force --per-country 120 --max-workers 4 --save-raw data/interim/qa/raw_wikidata.csv
      break
    } else {
      Write-Host "[watch] HTTP $($resp.StatusCode). Reintento en 60s..."
    }
  } catch {
    Write-Host "[watch] sin respuesta. Reintento en 60s..."
  }
  Start-Sleep -Seconds 60
}
