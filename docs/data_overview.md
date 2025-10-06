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
- Filas: 838
- Columnas: 12
- Columnas numéricas: company_id, signal_strength
- Columnas categóricas/texto: company_name, country, industry, size_bin, source, signal_type, ts, url, title, text_snippet

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_id | int64 | 838 | 0.00 | 266 | 6, 211, 32 |
| company_name | object | 838 | 0.00 | 268 | Ferrocarril de Bahía a São Francisco, Transporte Automotor La Plata S.A., Librerías de Cristal S. A. De C. V. |
| country | object | 838 | 0.00 | 7 | BR, AR, MX |
| industry | object | 838 | 0.00 | 12 | transport_logistics, retail_consumer, construction_realestate |
| size_bin | object | 838 | 0.00 | 2 | m, l |
| source | object | 838 | 0.00 | 2 | memberships, news |
| signal_type | object | 838 | 0.00 | 2 | sbti, news |
| signal_strength | float64 | 838 | 0.00 | 4 | 1.0, 0.5, 0.6 |
| ts | object | 838 | 0.00 | 528 | 2023-06-15 01:00:00, 2023-06-01 01:00:00, 2020-03-26 01:00:00 |
| url | object | 838 | 0.00 | 526 | https://files.sciencebasedtargets.org/production/files/companies-excel.xlsx, https://arstechnica.com/security/2025/09/supermicro-server-motherboards-can-b..., https://news.google.com/rss/articles/CBMisgFBVV95cUxNQmRZZ0l0RElsLWVySF9tbmpW... |
| title | object | 838 | 0.00 | 523 | Membership: sbti, Supermicro server motherboards can be infected with unremovable malware, Subte gratis para jubilados: cuáles son los requisitos y cómo hacer el trámit... |
| text_snippet | object | 838 | 0.00 | 838 | Citrosuco Agroindustria S.A., Topsoe A/S, A2A S.p.A. |

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
Listado preliminar de prospectos enfocados en renovables (fuentes climáticas, excluyendo la muestra base).

- Ruta: `data/processed/prospects_renewables.csv`
- Filas: 17
- Columnas: 4
- Columnas categóricas/texto: company_name, country, industry, source

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_name | object | 17 | 0.00 | 17 | Ambev S.A., Ammper Energía S.A.P.I de C.V, Chilexpress S.A. |
| country | object | 17 | 0.00 | 4 | BR, MX, CL |
| industry | object | 17 | 0.00 | 4 | agro_food, energy_power, transport_logistics |
| source | object | 17 | 0.00 | 1 | sbti |

## prospects_digital_assets.csv
Listado preliminar de prospectos enfocados en activos digitales/clima-tech (fuentes climáticas, excluyendo la muestra base).

- Ruta: `data/processed/prospects_digital_assets.csv`
- Filas: 21
- Columnas: 4
- Columnas categóricas/texto: company_name, country, industry, source

| columna | tipo | no nulos | % nulos | únicos | ejemplos |
|---|---|---:|---:|---:|---|
| company_name | object | 21 | 0.00 | 21 | Algar Telecom, America Movil, S.A.B. de C.V., Betterfly |
| country | object | 21 | 0.00 | 5 | BR, MX, CL |
| industry | object | 21 | 0.00 | 2 | ict_telecom, manufacturing |
| source | object | 21 | 0.00 | 1 | sbti |
