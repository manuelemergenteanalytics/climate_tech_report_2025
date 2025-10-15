# Resumen Estadístico de Datos

## universe_sample.csv
Marco muestral estratificado actualizado (Wikidata + Wikipedia) con umbral ≥100 empleados y filtros por industria.

- Ruta: `data/processed/universe_sample.csv`
- Filas: 1,200
- Columnas: 8
- Columnas numéricas: company_id, weight_stratum
- Columnas categóricas/texto: company_name, country, industry, size_bin, company_domain, ticker

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_id | int64 | 1200 | 0.00 | 1200 | 1, 2, 3 |
| company_name | object | 1200 | 0.00 | 1195 | Compañía de Minas Buenaventura S.A.A., Bodytech, Cervecería CCU Limache |
| country | object | 1200 | 0.00 | 7 | PE, BR, CL |
| industry | object | 1200 | 0.00 | 12 | mining_metals, ict_telecom, agro_food |
| size_bin | object | 1200 | 0.00 | 2 | l, m |
| company_domain | object | 397 | 66.92 | 377 | buenaventura.com, bodytech.com.br, cafetortoni.com.ar |
| weight_stratum | float64 | 1200 | 0.00 | 26 | 1.5584415584415583, 0.7792207792207793, 0.8202323991797676 |
| ticker | object | 26 | 97.83 | 25 | BVN@New York Stock Exchange, CENCOSUD@Santiago Stock Exchange, BSAC@New York Stock Exchange |

## events_normalized.csv
Señales normalizadas disponibles tras la ingesta más reciente (news + memberships).

- Ruta: `data/processed/events_normalized.csv`
- Filas: 6,561
- Columnas: 17
- Columnas numéricas: signal_strength, climate_score, sentiment_score
- Columnas categóricas/texto: company_id, company_name, country, industry, size_bin, source, signal_type, ts, url, title, text_snippet, description, sentiment_label

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_id | object | 6,561 | 0.00 | 5,680 | ext::ungc::natura-co-br, ext::sbti::acbel-polytech-tw, 102 |
| company_name | object | 6,561 | 0.00 | 5,612 | Natura & Co, COPEL- Companhia Paranaense de Energia, Samarco Mineracao S.A. |
| country | object | 6,561 | 0.00 | 18 | BR, CL, CO |
| industry | object | 6,561 | 0.00 | 15 | agro_food_beverage, energy_power_utilities, mining_materials |
| size_bin | object | 5,569 | 15.12 | 3 | l, m, s |
| source | object | 6,561 | 0.00 | 2 | memberships, news |
| signal_type | object | 6,561 | 0.00 | 4 | ungc, bcorp, sbti |
| signal_strength | float64 | 6,561 | 0.00 | 11 | 1.0, 0.18, -0.25 |
| ts | object | 6,561 | 0.00 | 3,387 | 2000-07-26T00:00:00+00:00, 2016-02-16T00:00:00+00:00, 2025-09-24T16:06:03+00:00 |
| url | object | 6,561 | 0.00 | 5,650 | https://unglobalcompact.org/…, https://files.sciencebasedtargets.org/…, https://news.google.com/… |
| title | object | 6,561 | 0.00 | 70 | Participación Pacto Global, Certificación B Corp, Compromiso SBTi |
| text_snippet | object | 6,561 | 0.00 | 5,649 | Natura & Co, COPEL- Companhia Paranaense de Energia, Topsoe A/S |
| description | object | 6,561 | 0.00 | 5,687 | Source=ungc | sector: Chemicals…, Source=ungc | sector: Electricity…, News: Ferrocarril reanuda operación |

## interest_scores.csv
Interés corporativo calculado (IIC) sobre la muestra representativa.

- Ruta: `data/processed/interest_scores.csv`
- Filas: 1,200
- Columnas: 9
- Columnas numéricas: company_id, group_memberships, signals_count, score_0_100
- Columnas categóricas/texto: company_name, country, industry, size_bin, last_ts

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_id | int64 | 1200 | 0.00 | 1200 | 32, 206, 6 |
| company_name | object | 1200 | 0.00 | 1195 | Tren de la República de los Niños, La Compañía, Café Tortoni |
| country | object | 1200 | 0.00 | 7 | AR, UY, BR |
| industry | object | 1200 | 0.00 | 12 | transport_logistics, agro_food, finance_insurance |
| size_bin | object | 1200 | 0.00 | 2 | m, l |
| group_memberships | float64 | 1200 | 0.00 | 8 | 50.0, 27.750514403490065, 21.164577638741143 |
| signals_count | int64 | 1200 | 0.00 | 8 | 7, 3, 44 |
| last_ts | object | 12 | 99.00 | 11 | 2025-07-10T01:00:00Z, 2025-08-28T01:00:00Z, 2025-10-02T01:00:00Z |
| score_0_100 | float64 | 1200 | 0.00 | 8 | 50.0, 27.75, 21.16 |

## prospects_renewables.csv
Listado preliminar de prospectos enfocados en renovables (señales externas a la muestra base, posterior a reclasificación LLM).

- Ruta: `data/processed/prospects_renewables.csv`
- Filas: 25
- Columnas: 5
- Columnas categóricas/texto: company_name, country, industry, source, justification

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_name | object | 25 | 0.00 | 25 | Órigo Energia, TECNOVIA Inovacao e Sustentabilidade LTDA, Holcim (Argentina) |
| country | object | 25 | 0.00 | 4 | BR, AR, MX |
| industry | object | 25 | 0.00 | 7 | agro_food_beverage, energy_power_utilities, environmental_circular_services |
| source | object | 25 | 0.00 | 4 | bcorp|ungc, ungc, news |
| justification | object | 25 | 0.00 | 25 | signals: bcorp, ungc | last_update: 2022-07-11 | …, signals: news | last_update: 2025-10-06 | … |

## prospects_digital_assets.csv
Listado preliminar de prospectos enfocados en activos digitales/clima-tech (señales externas a la muestra base, posterior a reclasificación LLM).

- Ruta: `data/processed/prospects_digital_assets.csv`
- Filas: 25
- Columnas: 5
- Columnas categóricas/texto: company_name, country, industry, source, justification

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_name | object | 25 | 0.00 | 25 | Mercado Libre, Grupo San Cristobal, BancoEstado Microempresas |
| country | object | 25 | 0.00 | 5 | AR, CL, BR |
| industry | object | 25 | 0.00 | 5 | ict_digital_media, finance_insurance_capital, professional_services_consulting |
| source | object | 25 | 0.00 | 7 | news|sbti, news|ungc, bcorp|ungc |
| justification | object | 25 | 0.00 | 25 | signals: news, sbti | last_update: 2025-05-24 | …, signals: bcorp, ungc | last_update: 2024-06-19 | … |

