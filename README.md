# Climate Tech Report 2025 (Emergente Analytics)

**Propósito**  
Detectar, medir y priorizar la demanda corporativa de soluciones climáticas y activos digitales en LATAM.  
El sistema produce:
- **Índice de Intensidad Corporativa (IIC)**: presencia relativa en señales históricas (12m).
- **Prospect Score (PS)**: momentum “hot” en los últimos 90 días, pensado para acciones de BD.

---

## Estratos y Ventanas

- **Países**: MX, BR, CO, CL, AR, UY
- **Industrias** (slugs canónicos, con `config/industry_map.yml` para alias):  
  `energy_power`, `oil_gas`, `mining_metals`, `chemicals_materials`,  
  `manufacturing`, `construction_realestate`, `transport_logistics`,  
  `agro_food`, `retail_consumer`, `water_waste_circularity`,  
  `finance_insurance`, `ict_telecom`
- **Size bin**: usado como covariable, no como estrato
- **Ventanas**:  
  - IIC = 12 meses  
  - PS = 90 días “hot”  
- **Pilares IIC**: H (Hiring), C (Commitments), F (Finance), P (Pilots), X (Press/News)

---

## Señales

Cada scraper trabaja **sólo sobre la muestra (`universe_sample.csv`)**, respeta `robots.txt`, aplica backoff + caching y produce filas normalizadas en `data/processed/events_normalized.csv`.

| Fuente       | Señales (`signal_type`)                    | Ejemplos de detección |
|--------------|--------------------------------------------|-----------------------|
| **Memberships** | `sbti`, `re100`, `sistemab`, `ungc`        | Sistemas voluntarios |
| **News/Press** | `newsroom`, `pilot_news`                   | Green bonds, SLL, pilotos, hidrógeno, CCUS, MRV, ISO14001, TCFD/TNFD, CDP, UNGC, B Corp |
| **Jobs**     | `job_posting`                               | Búsqueda Indeed/LinkedIn/Glassdoor (ES/PT/EN) |
| **Finance**  | `green_bond`, `sll`                         | Notas de prensa / bolsas locales |
| **Webscan**  | `web_esg`                                   | Densidad de términos en secciones Sustainability/ESG |

---

## Formato de Evento

Todos los scrapers devuelven el mismo esquema:

```text
company_id, company_name, country, industry, size_bin,
source, signal_type, signal_strength, ts, url, title, text_snippet
signal_strength: definido de forma reproducible (ej. recencia × densidad de keywords para News, seniority+frescura para Jobs).

dedupe: (company_id, signal_type, url) + canonicalización de URL.

Estructura del Repo
bash
Copiar código
climate_tech_report_2025/
├─ config/                # configs (industry_map.yml, keywords.yml, news.yml, …)
├─ data/
│  ├─ raw/                # dumps sin procesar
│  ├─ interim/            # cache y logs
│  └─ processed/          # universe_sample.csv, events_normalized.csv
├─ src/ctr25/
│  ├─ cli.py              # Typer CLI
│  ├─ signals/            # scrapers (news.py, jobs.py, finance.py, webscan.py)
│  ├─ sample_frame.py     # construcción de la muestra
│  ├─ iic.py              # cálculo IIC
│  ├─ prospect.py         # cálculo Prospect Score
│  ├─ visualize.py        # visualizaciones
│  └─ utils/              # helpers
Instalación
Requiere Python ≥ 3.10 (Windows/PowerShell):

powershell
Copiar código
# crear entorno
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel

# instalar en editable
.\.venv\Scripts\python.exe -m pip install -e .
Quickstart
powershell
Copiar código
# 1) inicializar carpeta y configs
.\.venv\Scripts\ctr25 init

# 2) construir marco muestral
.\.venv\Scripts\ctr25 sample

# 3) recolectar señales (ejemplos)
.\.venv\Scripts\ctr25 collect-memberships
.\.venv\Scripts\ctr25 collect-news --country AR --industry energy_power --max-companies 1
.\.venv\Scripts\ctr25 collect-jobs
.\.venv\Scripts\ctr25 collect-finance
.\.venv\Scripts\ctr25 collect-webscan

# 4) computar índices
.\.venv\Scripts\ctr25 compute-iic
.\.venv\Scripts\ctr25 compute-ps

# 5) visualizar y exportar
.\.venv\Scripts\ctr25 viz
.\.venv\Scripts\ctr25 export --out reports/collaborators/prospects.csv
Buenas prácticas técnicas
Caching: data/interim/cache/<fuente>/

Logs: por estrato (ej. logs/news_AR_energy_power.log.csv)

QA:

cobertura por estrato

% falsos positivos en keywords

totales consultadas, con señal, errores

Compatibilidad: correr todo en Windows/PowerShell, sin necesidad de activar venv manualmente.