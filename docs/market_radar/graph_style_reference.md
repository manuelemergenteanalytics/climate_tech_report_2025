# Referencia de estilo de gráficos – Market Radar LATAM 2025

## Notas compartidas
- Tema registrado en `scripts/market_radar_theme.py:1` (`apply_market_radar_theme`).
- Fuente `Josefin Sans`, texto base `#164233`, fondo `#f6fbf8`, área de gráfico `#ecf7f0`.
- Colorway fallback: `[#0b5d1e, #1e7633, #2f9147, #41aa5f, #55c278, #6fce8f, #8bd9a6, #a8e3bd, #c8edd5, #e4f7e9]`; escala secuencial verde `[[0.0,#f2fbf5],[0.2,#d5f1dd],[0.4,#a8e0bf],[0.6,#74cf9f],[0.8,#3bbd79],[1.0,#138f4c]]`.
- Hover labels: fondo `#123b2a`, borde `#0a251a`, texto `#f2fbf5`, 12px.
- `_write_plotly_html` inserta CSS uniforme y se eliminan los títulos internos (los encabezados vienen del layout principal).
- Industrias se renombran y envuelven mediante `apply_industry_labels` + `wrap_label`; colores fijos: `Agro & alimentos #1b9aaa`, `Construcción... #f4a259`, `Finanzas... #5d5179`, `TIC... #2ec4b6`, `Manufactura industrial #ff6f59`, `Petróleo & gas #d1495b`, `Retail... #9bc53d`, `Transporte... #3b8ea5`, `Energía & Servicios Públicos #00a896`, `Salud... #845ec2`, `Hospitalidad... #f6ae2d`, `Minería... #2a9d8f`, `Servicios profesionales... #33658a`, `Sector público... #ff9f1c`, `Servicios ambientales... #76b041`, `Sin industria #8bd9a6`.
- Países en el ranking usan tonos pastel: AR `#a4c8f0`, BR `#81c784`, CL `#ffe0a3`, CO `#9fd6d9`, MX `#ffb4a2`, PE `#b7d8a6`, UY `#cbb7e8`, VE `#ffe9b3`.

## Tablas (`coverage_summary.html`, `total_signal_summary.html`)
- `go.Table` 760×520 px, encabezados `#b8e3c6`, celdas `#e6f6ec`, líneas `#cfe8d7`, 14/13px.

## Sankey (`signal_mix_sankey.html`)
- Lienzo 860×560 px; nodos gradiente verde (`pad/thickness=18`, borde `#0a3c28`), enlaces RGBA coordinados.

## Ranking por industria (`coverage_indicators.html`)
- Barras 860×540 px (`#6ccfa0` + borde `#0f5134`) con conteo encima.
- Eje Y ordenado de mayor a menor gracias a `categoryarray` explícito.

## Heatmap país-industria (`coverage_country_industry.html`)
- 860×580 px, `xgap = ygap = 1`, escala secuencial verde, colorbar “# señales”. Industrias en castellano.

## Evolución temporal (`coverage_timeline.html`)
- Líneas 880×520 px; colores contrastados por industria, leyenda horizontal (`y=-0.18`).

## Brechas de cobertura (`coverage_gaps.html`)
- Dispersión 860×560 px, burbujas verde claro `#6ccfa0` (borde `#0a3c28`).
- Cada punto incluye una etiqueta multilínea visible usando `mode='markers+text'`; se posiciona sobre o bajo la burbuja según el caso (p.ej. `TIC, Digital & Medios` se alinea abajo) para evitar superposiciones.

## Heatmap de densidad (`heatmap_country_industry.html`)
- `px.density_heatmap` 860×580 px; colorbar “Σ score empresas”; eje X en industrias amigables.

## Cronología por tipo (`timeline_signals.html`)
- Barras agrupadas 880×500 px, coloreadas mediante `build_signal_type_color_map`.

## Choropleth (`map_intensity.html`)
- Mapa 880×640 px; proyección Mercator (`scale=1.2`, `lat[-58,33]`, `lon[-125,-32]`) cubriendo LATAM completa.
- Escala verde y colorbar “total_score”.

## Treemap (`country_fact_sheet.html`)
- 880×680 px, jerarquía país → industria con etiquetas envueltas (ancho 14 caracteres) y fuentes 12/10 para evitar desbordes.

## Ranking de compañías (`ranking_companies.html`)
- Barras horizontales 860×720 px, colores pastel por país (paleta superior) con borde `#0a3c28`, rango X ajustado (`min*0.85` – `max*1.05`).

## Tipos de señales (`signal_type_distribution.html`)
- Barras 780×450 px en `#62c68f` (borde `#0f5134`), ordenadas de mayor a menor.

## Operativa
- Gráficos ajustados al contenedor sin scroll horizontal.
- Regenerar tras cambios: `python3 scripts/generate_market_radar.py`.
