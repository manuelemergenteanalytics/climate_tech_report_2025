"""Build exploratory HTML visuals for the LATAM climate/digital demand radar."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import yaml
import numpy as np
import json
from uuid import uuid4

from market_radar_theme import (
    apply_market_radar_theme,
    MARKET_RADAR_INLINE_STYLE,
    MARKET_RADAR_SEQUENTIAL,
    MARKER_BORDER_COLOR,
    BAR_LINE_COLOR,
    COVERAGE_INDICATORS_BAR_COLOR,
    SIGNAL_TYPE_BAR_COLOR,
    COUNTRY_CATEGORY_ORDER,
    INDUSTRY_CATEGORY_ORDER,
    SIGNAL_TYPE_ORDER,
    build_country_color_map,
    build_industry_color_map,
    build_signal_type_color_map,
    ordered_categories,
    sankey_node_colors,
    sankey_link_colors,
    SANKEY_NODE_PAD,
    SANKEY_NODE_THICKNESS,
    SANKEY_NODE_LINE_COLOR,
    SANKEY_NODE_LINE_WIDTH,
    colorbar_defaults,
    PRIMARY_FONT,
    PRIMARY_TEXT_COLOR,
    friendly_industry_label,
    apply_industry_labels,
    wrap_label,
)

try:
    import networkx as nx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    nx = None

DATA_DIR = Path("data/processed")
EVENTS_PATH = DATA_DIR / "events_normalized.csv"
UNIVERSE_PATH = DATA_DIR / "universe_sample.csv"
KEYWORDS_PATH = Path("config/keywords.yml")
OUTPUT_DIR = Path("docs/market_radar/html")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP25_PATH = OUTPUT_DIR.parent / "top25_companies.txt"

PLOTLYJS_SCRIPT = (
    '<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.1.0.min.js" '
    'integrity="sha256-Ei4740bWZhaUTQuD6q9yQlgVCMPBz6CZWhevDYPv93A=" crossorigin="anonymous"></script>'
)

ISO2_TO_ISO3 = {
    "AR": "ARG",
    "BO": "BOL",
    "BR": "BRA",
    "CL": "CHL",
    "CO": "COL",
    "CR": "CRI",
    "DO": "DOM",
    "EC": "ECU",
    "GT": "GTM",
    "HN": "HND",
    "MX": "MEX",
    "NI": "NIC",
    "PA": "PAN",
    "PE": "PER",
    "PY": "PRY",
    "SV": "SLV",
    "UY": "URY",
    "VE": "VEN",
}

FOCUS_COUNTRIES = {"AR", "BR", "CL", "CO", "PE", "UY", "MX"}

INDUSTRY_SIMPLIFY = {
    "agro_food_beverage": "agro_food",
    "agro_food": "agro_food",
    "construction_infrastructure_realestate": "construction_realestate",
    "energy_power_utilities": "energy_power",
    "finance_insurance_capital": "finance_insurance",
    "ict_digital_media": "ict_telecom",
    "industrial_manufacturing": "manufacturing",
    "mining_materials": "mining_metals",
    "retail_consumer_services": "retail_consumer",
    "transport_mobility_logistics": "transport_logistics",
    "oil_gas": "oil_gas",
    "water_waste_circularity": "water_waste_circularity",
    "chemicals_materials": "chemicals_materials",
}


apply_market_radar_theme()


def _load_keywords(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    words = [str(w).strip() for w in data.get("include", []) if str(w).strip()]
    # remove duplicates while keeping order
    seen = set()
    ordered: List[str] = []
    for word in words:
        key = word.lower()
        if key in seen:
            continue
        ordered.append(word)
        seen.add(key)
    return ordered


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _weighted_strength(df: pd.DataFrame) -> pd.Series:
    strength = pd.to_numeric(df.get("signal_strength"), errors="coerce")
    climate = pd.to_numeric(df.get("climate_score"), errors="coerce")
    return (strength.fillna(0.0) * climate.fillna(1.0)).fillna(0.0)


def _prep_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_PATH)
    events["ts"] = pd.to_datetime(events["ts"], errors="coerce", utc=True)
    events["event_date"] = events["ts"].dt.date
    events["weighted_strength"] = _weighted_strength(events)
    events["abs_strength"] = events["weighted_strength"].abs()
    events["sentiment_score"] = pd.to_numeric(events.get("sentiment_score"), errors="coerce")
    events["signal_strength"] = pd.to_numeric(events.get("signal_strength"), errors="coerce")
    events["climate_score"] = pd.to_numeric(events.get("climate_score"), errors="coerce")
    raw_ids = events.get("company_id")
    events["company_id"] = raw_ids.astype(str) if raw_ids is not None else ""
    events["company_id"] = events["company_id"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    events["company_numeric_id"] = pd.to_numeric(raw_ids, errors="coerce")
    events["company_key"] = events["company_id"].copy()
    missing_key = events["company_key"].isna()
    if missing_key.any():
        fallback_qid = events.loc[missing_key, "company_qid"].astype(str).replace({"nan": pd.NA, "None": pd.NA})
        events.loc[missing_key, "company_key"] = fallback_qid
    still_missing = events["company_key"].isna()
    if still_missing.any():
        fallback_name = events.loc[still_missing, "company_name"].fillna("")
        events.loc[still_missing, "company_key"] = fallback_name.str.lower().str.replace(r"\s+", "_", regex=True)
    events["company_key"] = events["company_key"].fillna("unknown_company")
    events["country"] = events["country"].fillna("").astype(str).str.upper()
    events["industry"] = events["industry"].fillna("").astype(str)
    events["signal_type"] = events["signal_type"].fillna("unknown")
    events["source"] = events["source"].fillna("unknown")
    return events


def _company_rollup(events: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events.groupby(
            [
                "company_key",
                "company_name",
                "country",
                "industry",
            ],
            as_index=False,
        )
        .agg(
            avg_strength=("weighted_strength", "mean"),
            avg_abs_strength=("abs_strength", "mean"),
            total_events=("weighted_strength", "count"),
            last_ts=("ts", "max"),
            avg_sentiment=("sentiment_score", "mean"),
            company_id=("company_id", "first"),
            company_numeric_id=("company_numeric_id", "first"),
            company_qid=("company_qid", "first"),
        )
    )
    grouped["total_events"] = grouped["total_events"].astype(int)
    grouped["last_ts"] = pd.to_datetime(grouped["last_ts"], errors="coerce", utc=True)
    boost = (1 + 0.15 * np.log1p(grouped["total_events"])).clip(upper=1.8)
    grouped["demand_score"] = grouped["avg_strength"] * boost
    grouped["intensity_score"] = grouped["avg_abs_strength"] * boost
    return grouped


def _prep_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["company_id"] = pd.to_numeric(universe["company_id"], errors="coerce")
    universe["industry"] = universe["industry"].fillna("")
    universe["country"] = universe["country"].fillna("")
    return universe


def _filter_events_by_universe(events: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Keep only events whose company belongs to the provided universe."""

    if universe.empty:
        return events.iloc[0:0].copy()

    universe_ids = pd.to_numeric(universe.get("company_id"), errors="coerce").dropna()
    if universe_ids.empty:
        return events.iloc[0:0].copy()

    universe_set = set(universe_ids)
    company_ids = pd.to_numeric(events.get("company_numeric_id"), errors="coerce")
    mask = company_ids.isin(universe_set)
    return events[mask].copy()


def _clean_industry(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.where(cleaned.ne(""), "Sin industria")


def _align_industry(series: pd.Series) -> pd.Series:
    cleaned = _clean_industry(series)
    mapped = cleaned.map(INDUSTRY_SIMPLIFY)
    return mapped.fillna(cleaned)


def heatmap_country_industry(companies: pd.DataFrame) -> Path:
    pivot = (
        companies.groupby(["country", "industry"], as_index=False)
        .agg(total_score=("demand_score", "sum"), companies=("company_id", "count"))
    )
    if pivot.empty:
        pivot = pd.DataFrame({"country": [], "industry": [], "total_score": []})

    pivot["industry_display"] = apply_industry_labels(pivot["industry"])

    fig = px.density_heatmap(
        pivot,
        x="industry_display",
        y="country",
        z="total_score",
        color_continuous_scale=[step[1] for step in MARKET_RADAR_SEQUENTIAL],
        title="Intensidad de señales por país e industria",
        labels={"total_score": "Σ score empresas", "industry_display": "Industria"},
    )
    fig.update_coloraxes(
        colorscale=MARKET_RADAR_SEQUENTIAL, colorbar=colorbar_defaults("Σ score empresas")
    )
    fig.update_layout(title=None, width=860, height=580, yaxis=dict(categoryorder="total ascending"))

    path = OUTPUT_DIR / "heatmap_country_industry.html"
    return _write_plotly_html(fig, path)


def timeline_signals(events: pd.DataFrame) -> Path:
    events = events.copy()
    events["month"] = events["ts"].dt.to_period("M").dt.to_timestamp()
    company_month = (
        events.groupby(["month", "signal_type", "company_id"], as_index=False)
        .agg(strength=("weighted_strength", "mean"))
    )
    timeline = (
        company_month.groupby(["month", "signal_type"], as_index=False)
        .agg(companies=("company_id", "nunique"), strength=("strength", "sum"))
    )
    if timeline.empty:
        timeline = pd.DataFrame({"month": [], "signal_type": [], "companies": [], "strength": []})

    color_map = build_signal_type_color_map(timeline["signal_type"].unique())
    order = ordered_categories(timeline["signal_type"], SIGNAL_TYPE_ORDER)

    fig = px.bar(
        timeline,
        x="month",
        y="strength",
        color="signal_type",
        barmode="group",
        title="Cronología de señales por tipo",
        labels={"strength": "Σ score empresas", "month": "Mes"},
        color_discrete_map=color_map,
        category_orders={"signal_type": order},
    )
    fig.update_layout(title=None, width=880, height=500, bargap=0.15)

    path = OUTPUT_DIR / "timeline_signals.html"
    return _write_plotly_html(fig, path)


def coverage_timeline(events: pd.DataFrame) -> Path:
    events = events.copy()
    if events.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Evolución mensual de señales",
            width=950,
            height=520,
            annotations=[
                dict(
                    text="Sin datos disponibles",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        path = OUTPUT_DIR / "coverage_timeline.html"
        return _write_plotly_html(fig, path)

    events["month"] = events["ts"].dt.to_period("M").dt.to_timestamp()
    events["industry_clean"] = _clean_industry(events.get("industry"))

    monthly = (
        events.groupby(["month", "industry_clean"], as_index=False)
        .agg(
            total_signals=("company_key", "count"),
            total_strength=("weighted_strength", "sum"),
        )
    )
    if monthly.empty:
        monthly = pd.DataFrame(
            {"month": [], "industry_clean": [], "total_signals": [], "total_strength": []}
        )

    if not monthly.empty:
        latest = monthly["month"].max()
        cutoff = latest - pd.DateOffset(months=23)
        monthly = monthly[monthly["month"] >= cutoff]

    industry_order = (
        monthly.groupby("industry_clean")["total_signals"].sum().sort_values(ascending=False)
    )
    top_industries = industry_order.head(8).index.tolist()
    filtered = monthly[monthly["industry_clean"].isin(top_industries)].copy()

    if filtered.empty:
        filtered = monthly.copy()

    filtered.sort_values("month", inplace=True)
    filtered["industry_display"] = apply_industry_labels(filtered["industry_clean"])

    industry_color_map = build_industry_color_map(filtered["industry_display"].unique())
    legend_order = ordered_categories(filtered["industry_display"], INDUSTRY_CATEGORY_ORDER)

    fig = px.line(
        filtered,
        x="month",
        y="total_signals",
        color="industry_display",
        markers=True,
        title="Evolución mensual de señales por industria (Top 8)",
        labels={
            "month": "Mes",
            "total_signals": "# de señales",
            "industry_display": "Industria",
        },
        hover_data={"total_strength": ":.2f"},
        color_discrete_map=industry_color_map,
        category_orders={"industry_display": legend_order},
    )
    fig.update_layout(
        title=None,
        width=880,
        height=520,
        legend=dict(
            title=dict(text="Industria", font=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=14)),
            orientation="h",
            y=-0.18,
            font=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=12),
        ),
    )

    path = OUTPUT_DIR / "coverage_timeline.html"
    return _write_plotly_html(fig, path)


def map_intensity(companies: pd.DataFrame) -> Path:
    geo = (
        companies.groupby("country", as_index=False)
        .agg(total_score=("demand_score", "sum"), companies=("company_id", "count"))
    )
    geo["iso3"] = geo["country"].str.upper().map(ISO2_TO_ISO3)
    fig = px.choropleth(
        geo,
        locations="iso3",
        locationmode="ISO-3",
        color="total_score",
        hover_data={"companies": True, "total_score": ":.2f"},
        title="Intensidad de señales climáticas/digitales por país",
        color_continuous_scale=[step[1] for step in MARKET_RADAR_SEQUENTIAL],
    )
    fig.update_layout(title=None, width=880, height=640, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_coloraxes(colorscale=MARKET_RADAR_SEQUENTIAL, colorbar=colorbar_defaults("total_score"))
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=-18, lon=-70),
        projection_scale=1.2,
        lataxis=dict(range=[-58, 33]),
        lonaxis=dict(range=[-125, -32]),
        showcountries=True,
        countrycolor="#3b3b3b",
        showsubunits=True,
        subunitcolor="#ffffff",
        landcolor="#d4f0e0",
        lakecolor="#f3f9f4",
    )
    path = OUTPUT_DIR / "map_intensity.html"
    return _write_plotly_html(fig, path)


def _extract_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    lowered = text.lower()
    hits = []
    for kw in keywords:
        base = kw.lower()
        if base in lowered:
            hits.append(kw)
    return hits


def network_themes(events: pd.DataFrame, keywords: List[str]) -> List[Path]:
    news = events[events["signal_type"].eq("news")].copy()
    news["text_all"] = news[["title", "text_snippet"]].fillna("").agg(" ".join, axis=1)
    news["matched"] = news["text_all"].apply(lambda txt: sorted(set(_extract_keywords(txt, keywords))))
    edges: Dict[Tuple[str, str], int] = {}
    for items in news["matched"]:
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = tuple(sorted((items[i], items[j])))
                edges[pair] = edges.get(pair, 0) + 1

    # Heatmap representation (always generated)
    matrix: Dict[str, Dict[str, int]] = {kw: {kw2: 0 for kw2 in keywords} for kw in keywords}
    for (a, b), weight in edges.items():
        matrix[a][b] = weight
        matrix[b][a] = weight
    df_matrix = pd.DataFrame(matrix)
    if df_matrix.empty:
        df_matrix = pd.DataFrame({"sin datos": [0]})
    heatmap_fig = px.imshow(
        df_matrix,
        color_continuous_scale="Purples",
        title="Co-ocurrencia de temas climáticos en noticias",
        labels=dict(color="Frecuencia conjunta"),
    )
    heatmap_fig.update_layout(width=900, height=650)
    heatmap_path = OUTPUT_DIR / "network_themes_heatmap.html"
    heatmap_fig.write_html(heatmap_path, include_plotlyjs="cdn")

    paths = [heatmap_path]

    if edges and nx is not None:
        G = nx.Graph()
        for kw in keywords:
            G.add_node(kw)
        for (a, b), weight in edges.items():
            if weight > 0:
                G.add_edge(a, b, weight=weight)

        nodes_to_keep = {n for n, d in G.degree() if d > 0}
        H = G.subgraph(nodes_to_keep).copy()
        if H.number_of_nodes() == 0:
            H.add_node("sin datos")

        pos = nx.spring_layout(H, seed=42, k=0.6)
        edge_x: List[float] = []
        edge_y: List[float] = []
        for u, v in H.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        text = []
        marker_size = []
        for node in H.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            degree = H.degree(node)
            marker_size.append(12 + degree * 3)
            text.append(f"{node} (grado {degree})")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[n for n in H.nodes()],
            textposition="top center",
            hovertext=text,
            hoverinfo="text",
            marker=dict(
                showscale=False,
                size=marker_size,
                color="#0b7285",
                line=dict(width=1, color="#063c4d"),
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Red de temas climáticos en noticias",
            showlegend=False,
            width=900,
            height=650,
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        graph_path = OUTPUT_DIR / "network_themes_graph.html"
        fig.write_html(graph_path, include_plotlyjs="cdn")
        paths.append(graph_path)

    return paths


def _json_to_tabs(obj: Any) -> str:
    text = json.dumps(obj, indent=2, ensure_ascii=False, cls=PlotlyJSONEncoder)
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        spaces = len(line) - len(stripped)
        tabs = "\t" * (spaces // 2)
        lines.append(f"{tabs}{stripped}")
    return "\n".join(lines)


def _write_plotly_html(fig: go.Figure, path: Path, config: Dict[str, Any] | None = None) -> Path:
    cfg: Dict[str, Any] = {"responsive": True}
    if config:
        cfg.update(config)

    plotly_json = fig.to_plotly_json()
    div_id = f"plotly-{uuid4().hex}"
    height = fig.layout.height or 600
    width = fig.layout.width or 900
    data_json = _json_to_tabs(plotly_json.get("data", []))
    layout_json = _json_to_tabs(plotly_json.get("layout", {}))
    config_json = _json_to_tabs(cfg)

    style_lines = [line for line in MARKET_RADAR_INLINE_STYLE.strip().splitlines() if line]

    html_lines = [
        "<html>",
        "<head>",
        "\t<meta charset=\"utf-8\" />",
        "\t<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>",
        f"\t{PLOTLYJS_SCRIPT}",
        "\t<style>",
    ]
    html_lines.extend(f"\t{line}" for line in style_lines)
    html_lines.extend([
        "\t</style>",
        "</head>",
        "<body>",
        f"\t<div id=\"{div_id}\" class=\"plotly-graph-div\" style=\"height:{height}px; width:{width}px;\"></div>",
        "\t<script type=\"text/javascript\">",
        "\t\twindow.PLOTLYENV=window.PLOTLYENV || {};",
        f"\t\tif (document.getElementById(\"{div_id}\")) {{",
        "\t\t\tPlotly.newPlot(",
        f"\t\t\t\t\"{div_id}\",",
        "\t\t\t\t" + data_json.replace("\n", "\n\t\t\t\t") + ",",
        "\t\t\t\t" + layout_json.replace("\n", "\n\t\t\t\t") + ",",
        "\t\t\t\t" + config_json.replace("\n", "\n\t\t\t\t"),
        "\t\t\t);",
        "\t\t}",
        "\t</script>",
        "</body>",
        "</html>",
    ])

    path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    return path


def ranking_companies(companies: pd.DataFrame) -> Path:
    ranking = (
        companies.sort_values("demand_score", ascending=False)
        .head(25)
        .copy()
    )
    ranking["last_ts_fmt"] = ranking["last_ts"].dt.strftime("%Y-%m-%d").fillna("—")
    ranking["industry_display"] = apply_industry_labels(ranking["industry"])

    country_values = ranking["country"].dropna().astype(str)
    color_map = build_country_color_map(country_values)
    country_order = ordered_categories(country_values, COUNTRY_CATEGORY_ORDER)

    fig = px.bar(
        ranking,
        x="demand_score",
        y="company_name",
        orientation="h",
        color="country",
        text="total_events",
        hover_data={"last_ts_fmt": True, "industry": True, "avg_strength": ":.2f"},
        title="Top 25 Empresas por Intensidad de Demanda",
        labels={"demand_score": "Score ponderado", "total_events": "# señales", "last_ts_fmt": "Última señal"},
        color_discrete_map=color_map,
        category_orders={"country": country_order},
    )
    fig.update_traces(marker=dict(line=dict(color=MARKER_BORDER_COLOR, width=0.8)))
    fig.update_traces(textfont=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=12))
    min_score = float(ranking["demand_score"].min()) if not ranking.empty else 0.0
    max_score = float(ranking["demand_score"].max()) if not ranking.empty else 0.0
    lower_bound = 0.0 if min_score <= 0 else min_score * 0.85
    upper_bound = max_score * 1.05 if max_score else 1.0

    fig.update_layout(
        title=None,
        height=720,
        width=860,
        yaxis=dict(automargin=True),
        legend=dict(
            title=dict(text="País", font=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=14)),
            font=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=14),
        ),
    )
    fig.update_xaxes(range=[lower_bound, upper_bound])

    # Export plain text summary
    lines = ["Top 25 Empresas por Intensidad de Demanda\n"]
    for idx, row in ranking.iterrows():
        score = f"{row['demand_score']:.2f}"
        avg = f"{row['avg_strength']:.2f}"
        lines.append(
            f"- {row['company_name']} ({row['country']} · {row['industry']}): score {score}, promedio {avg}, señales {row['total_events']}, última {row['last_ts_fmt']}"
        )
    TOP25_PATH.write_text("\n".join(lines), encoding="utf-8")

    path = OUTPUT_DIR / "ranking_companies.html"
    return _write_plotly_html(fig, path)


def sentiment_distribution(companies: pd.DataFrame) -> Path:
    def _label(score: float) -> str:
        if pd.isna(score):
            return "neutral"
        if score >= 0.2:
            return "positive"
        if score <= -0.2:
            return "negative"
        return "neutral"

    tmp = companies.copy()
    tmp["sentiment_label"] = tmp["avg_sentiment"].apply(_label)
    agg = (
        tmp.groupby(["industry", "sentiment_label"], as_index=False)
        .agg(companies=("company_id", "count"))
    )
    fig = px.bar(
        agg,
        x="industry",
        y="companies",
        color="sentiment_label",
        barmode="stack",
        title="Distribución de sentimiento por industria",
        labels={"companies": "Número de empresas"},
    )
    fig.update_layout(width=950, height=500, xaxis=dict(automargin=True))

    path = OUTPUT_DIR / "sentiment_distribution.html"
    return _write_plotly_html(fig, path)


def coverage_indicators(companies: pd.DataFrame) -> Path:
    rollup = (
        companies.groupby("industry", as_index=False)
        .agg(
            total_companies=("company_key", "nunique"),
            total_events=("total_events", "sum"),
            total_demand=("demand_score", "sum"),
            median_intensity=("intensity_score", "median"),
        )
    )
    rollup = rollup[rollup["total_companies"] > 0]
    rollup = rollup.sort_values("total_demand", ascending=False).head(20)
    rollup["avg_events_per_company"] = (rollup["total_events"] / rollup["total_companies"]).round(2)
    rollup["industry_friendly"] = apply_industry_labels(rollup["industry"])

    ordered_by_demand = rollup.sort_values("total_demand", ascending=False)[
        "industry_friendly"
    ].tolist()
    industry_order = ordered_categories(rollup["industry_friendly"], INDUSTRY_CATEGORY_ORDER)
    category_array = [item for item in ordered_by_demand if item in industry_order]

    fig = px.bar(
        rollup,
        x="total_demand",
        y="industry_friendly",
        orientation="h",
        text="total_companies",
        title="Industrias con mayor intensidad de señales",
        labels={
            "total_demand": "Σ score ponderado",
            "industry_friendly": "Industria",
            "total_companies": "Empresas",
        },
        hover_data={
            "total_events": True,
            "median_intensity": ":.2f",
            "avg_events_per_company": True,
        },
        color_discrete_sequence=[COVERAGE_INDICATORS_BAR_COLOR],
        category_orders={"industry_friendly": industry_order},
    )
    fig.update_traces(
        marker=dict(color=COVERAGE_INDICATORS_BAR_COLOR, line=dict(color=BAR_LINE_COLOR, width=0.6))
    )
    fig.update_layout(
        title=None,
        width=860,
        height=540,
        yaxis=dict(
            automargin=True,
            categoryorder="array",
            categoryarray=category_array,
        ),
    )

    path = OUTPUT_DIR / "coverage_indicators.html"
    return _write_plotly_html(fig, path)


def coverage_country_industry(events: pd.DataFrame) -> Path:
    coverage = (
        events.groupby(["country", "industry"], as_index=False)
        .agg(
            total_events=("company_key", "count"),
            companies=("company_key", "nunique"),
            total_demand=("weighted_strength", "sum"),
        )
    )
    coverage = coverage[coverage["total_events"] > 0]
    path = OUTPUT_DIR / "coverage_country_industry.html"
    if coverage.empty:
        fig = go.Figure()
        fig.update_layout(
            title=None,
            width=860,
            height=580,
            annotations=[
                dict(
                    text="Sin datos disponibles",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        return _write_plotly_html(fig, path)

    matrix_events = coverage.pivot(index="country", columns="industry", values="total_events").fillna(0)
    countries = matrix_events.index.tolist()
    industries = matrix_events.columns.tolist()
    industry_map = {name: friendly_industry_label(name) for name in industries}
    matrix_events = matrix_events.rename(columns=industry_map)
    industries = matrix_events.columns.tolist()
    z_values = matrix_events.to_numpy()

    companies_matrix = (
        coverage.pivot(index="country", columns="industry", values="companies")
        .reindex(index=countries, columns=industry_map.keys())
        .rename(columns=industry_map)
        .fillna(0)
    )
    demand_matrix = (
        coverage.pivot(index="country", columns="industry", values="total_demand")
        .reindex(index=countries, columns=industry_map.keys())
        .rename(columns=industry_map)
        .fillna(0.0)
    )

    custom = np.stack([companies_matrix.to_numpy(), demand_matrix.to_numpy()], axis=-1)

    heatmap = go.Heatmap(
        z=z_values,
        x=industries,
        y=countries,
        colorscale=MARKET_RADAR_SEQUENTIAL,
        colorbar=colorbar_defaults("# señales"),
        customdata=custom,
        hovertemplate=(
            "industria=%{x}<br>país=%{y}<br>señales=%{z:.0f}<br>"
            "empresas únicas=%{customdata[0]:.0f}<br>Σ score=%{customdata[1]:.2f}<extra></extra>"
        ),
        xgap=1,
        ygap=1,
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=None,
        width=860,
        height=580,
        xaxis=dict(automargin=True, showgrid=False, zeroline=False),
        yaxis=dict(automargin=True, showgrid=False, zeroline=False),
    )

    return _write_plotly_html(fig, path)


def signal_type_distribution(events: pd.DataFrame) -> Path:
    counts = (
        events.groupby("signal_type", dropna=False)
        .size()
        .reset_index(name="total_signals")
    )
    counts["signal_type"] = counts["signal_type"].fillna("sin dato").astype(str)
    counts = counts.sort_values("total_signals", ascending=False)

    signal_order = counts.sort_values("total_signals", ascending=False)["signal_type"].tolist()

    fig = px.bar(
        counts,
        x="signal_type",
        y="total_signals",
        text="total_signals",
        title="Distribución de señales por tipo",
        labels={"signal_type": "Tipo de señal", "total_signals": "Cantidad"},
        category_orders={"signal_type": signal_order},
        color_discrete_sequence=[SIGNAL_TYPE_BAR_COLOR],
    )
    fig.update_traces(
        marker=dict(color=SIGNAL_TYPE_BAR_COLOR, line=dict(color=BAR_LINE_COLOR, width=0.6))
    )
    fig.update_layout(title=None, width=780, height=450, xaxis=dict(categoryorder="array", categoryarray=signal_order))

    path = OUTPUT_DIR / "signal_type_distribution.html"
    return _write_plotly_html(fig, path)


def coverage_gaps(companies: pd.DataFrame, universe: pd.DataFrame) -> Path:
    universe = universe.copy()
    universe["industry_clean"] = _align_industry(universe.get("industry"))
    universe_ids = pd.to_numeric(universe.get("company_id"), errors="coerce")
    universe["company_id_numeric"] = universe_ids

    universe_counts = (
        universe.dropna(subset=["company_id_numeric"])
        .groupby("industry_clean")
        .agg(universe_companies=("company_id_numeric", "nunique"))
    )

    companies = companies.copy()
    companies["industry_clean"] = _align_industry(companies.get("industry"))

    if companies.empty or universe_counts.empty:
        fig = go.Figure()
        fig.update_layout(
            title=None,
            width=860,
            height=560,
            annotations=[
                dict(
                    text="Sin datos suficientes para calcular brechas",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        path = OUTPUT_DIR / "coverage_gaps.html"
        return _write_plotly_html(fig, path)

    active = (
        companies.groupby("industry_clean", as_index=False)
        .agg(
            active_companies=("company_key", "nunique"),
            total_signals=("total_events", "sum"),
            total_demand=("demand_score", "sum"),
            median_intensity=("intensity_score", "median"),
        )
    )

    merged = active.merge(
        universe_counts.reset_index(),
        on="industry_clean",
        how="left",
    )
    merged = merged[merged["universe_companies"].notna()].copy()
    merged["coverage_pct"] = (
        merged["active_companies"] / merged["universe_companies"] * 100
    ).fillna(0.0)
    merged["avg_signals_per_company"] = (
        merged["total_signals"] / merged["active_companies"].replace(0, pd.NA)
    ).fillna(0.0)

    merged = merged[merged["universe_companies"] > 0]

    if merged.empty:
        fig = go.Figure()
        fig.update_layout(
            title=None,
            width=860,
            height=560,
            annotations=[
                dict(
                    text="Sin datos suficientes para calcular brechas",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        path = OUTPUT_DIR / "coverage_gaps.html"
        return _write_plotly_html(fig, path)

    merged["industry_display"] = apply_industry_labels(merged["industry_clean"])
    merged["industry_text"] = merged["industry_display"].apply(lambda val: wrap_label(val, width=18))
    text_positions = merged["industry_display"].apply(
        lambda name: "bottom center" if name == "TIC, Digital & Medios" else "top center"
    )

    fig = px.scatter(
        merged,
        x="coverage_pct",
        y="total_demand",
        size="avg_signals_per_company",
        hover_name="industry_display",
        labels={
            "coverage_pct": "% de universo con señales",
            "total_demand": "Σ score ponderado",
            "avg_signals_per_company": "Señales promedio por empresa",
        },
        hover_data={
            "median_intensity": ":.2f",
            "active_companies": True,
            "universe_companies": True,
        },
        title="Brechas de cobertura por industria",
    )
    fig.update_traces(
        mode="markers+text",
        marker=dict(color="#6ccfa0", opacity=0.85, line=dict(width=1, color=MARKER_BORDER_COLOR)),
        text=merged["industry_text"],
        textposition=text_positions,
        textfont=dict(color=PRIMARY_TEXT_COLOR, family=PRIMARY_FONT, size=10),
        cliponaxis=False,
    )
    fig.update_layout(
        title=None,
        width=860,
        height=560,
        xaxis=dict(range=[0, max(merged["coverage_pct"].max() * 1.1, 5)], zeroline=False),
        yaxis=dict(zeroline=False),
        showlegend=False,
        margin=dict(b=80, t=40, l=60, r=40),
    )

    path = OUTPUT_DIR / "coverage_gaps.html"
    return _write_plotly_html(fig, path)


def coverage_summary(
    companies: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    *,
    title: str = "Resumen estadístico de cobertura",
    filename: str = "coverage_summary.html",
) -> Path:
    total_events_series = pd.to_numeric(companies.get("total_events"), errors="coerce").fillna(0)

    company_numeric_series = pd.to_numeric(companies.get("company_numeric_id"), errors="coerce")
    fallback_series = companies.get("company_key")
    if fallback_series is None:
        fallback_series = pd.Series(index=companies.index, dtype="object")
    fallback_series = fallback_series.fillna("").astype(str)

    summary_ids = pd.Series(index=companies.index, dtype="object")
    numeric_mask = company_numeric_series.notna()
    if numeric_mask.any():
        summary_ids.loc[numeric_mask] = (
            company_numeric_series.loc[numeric_mask].astype("Int64").astype(str)
        )
    if (~numeric_mask).any():
        summary_ids.loc[~numeric_mask] = fallback_series.loc[~numeric_mask]
    summary_ids = summary_ids.replace("", "unknown_company").fillna("unknown_company")

    if total_events_series.empty or summary_ids.empty:
        events_per_company = np.array([])
    else:
        events_per_company = total_events_series.groupby(summary_ids).sum().to_numpy()

    total_signals = int(events_per_company.sum()) if events_per_company.size else 0

    numeric_ids = set(company_numeric_series.dropna().astype("Int64").tolist())

    if events_per_company.size:
        mean_signals = float(events_per_company.mean())
        median_signals = float(np.median(events_per_company))
        std_signals = float(events_per_company.std(ddof=0))
        q1 = float(np.percentile(events_per_company, 25))
        q3 = float(np.percentile(events_per_company, 75))
        max_signals = int(events_per_company.max())
        min_signals = int(events_per_company.min())
    else:
        mean_signals = median_signals = std_signals = q1 = q3 = 0.0
        max_signals = min_signals = 0

    metrics: List[Tuple[str, str]] = []

    if universe is not None and not universe.empty:
        universe_ids = pd.to_numeric(universe.get("company_id"), errors="coerce").dropna().astype("Int64")
        universe_set = set(universe_ids.tolist())
        total_universe = int(len(universe_set))

        active_in_universe = int(len(numeric_ids & universe_set))
        inactive_in_universe = max(total_universe - active_in_universe, 0)
        coverage_pct = (active_in_universe / total_universe * 100) if total_universe else 0.0

        metrics.extend(
            [
                ("Empresas en el universo", f"{total_universe:,}"),
                ("Empresas del universo con señales", f"{active_in_universe:,}"),
                ("Empresas del universo sin señales", f"{inactive_in_universe:,}"),
                ("% universo cubierto", f"{coverage_pct:.1f}%"),
            ]
        )

    unique_companies = int(summary_ids.nunique()) if not summary_ids.empty else 0

    metrics.extend(
        [
            ("Empresas únicas en la muestra", f"{unique_companies:,}"),
            ("Total de señales registradas", f"{total_signals:,}"),
            ("Señales por empresa (media)", f"{mean_signals:.2f}"),
            ("Señales por empresa (mediana)", f"{median_signals:.2f}"),
            ("Señales por empresa (desvío estándar)", f"{std_signals:.2f}"),
            ("Señales por empresa (Q1)", f"{q1:.2f}"),
            ("Señales por empresa (Q3)", f"{q3:.2f}"),
            ("Señales por empresa (mínimo)", f"{min_signals:,}"),
            ("Señales por empresa (máximo)", f"{max_signals:,}"),
        ]
    )

    labels = [metric for metric, _ in metrics]
    values = [value for _, value in metrics]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Métrica", "Valor"], align="left"),
                cells=dict(values=[labels, values], align="left"),
            )
        ]
    )
    fig.update_layout(title=None, width=760, height=520)

    path = OUTPUT_DIR / filename
    return _write_plotly_html(fig, path)


def total_signal_summary(companies: pd.DataFrame) -> Path:
    return coverage_summary(
        companies,
        universe=None,
        title="Resumen estadístico de señales (total)",
        filename="total_signal_summary.html",
    )


def signal_mix_sankey(events: pd.DataFrame, universe: pd.DataFrame) -> Path:
    universe_ids = pd.to_numeric(universe["company_id"], errors="coerce").dropna()
    universe_set = set(universe_ids)

    events_subset = events[events["company_numeric_id"].isin(universe_set)].copy()
    events_subset["signal_type"] = events_subset["signal_type"].fillna("unknown").astype(str)
    events_subset["company_numeric_id"] = pd.to_numeric(
        events_subset["company_numeric_id"], errors="coerce"
    )
    events_subset = events_subset.dropna(subset=["company_numeric_id"])

    active_ids = events_subset["company_numeric_id"].dropna().unique()
    active_count = int(active_ids.size)
    total_universe = int(len(universe_set))
    inactive_count = max(total_universe - active_count, 0)

    def _label_combo(series: pd.Series) -> str:
        unique = sorted({str(item) for item in series if pd.notna(item)})
        if not unique:
            return "sin tipo"
        return " + ".join(unique)

    combo_counts = (
        events_subset.groupby("company_numeric_id")["signal_type"].apply(_label_combo).value_counts()
    )

    if combo_counts.empty:
        combo_counts = pd.Series({"sin datos": 0})

    labels = ["Universo total", "Con señales", "Sin señales"] + combo_counts.index.tolist()
    sources = [0, 0]
    targets = [1, 2]
    values = [active_count, inactive_count]

    offset = 3
    for idx, (_, value) in enumerate(combo_counts.items()):
        sources.append(1)
        targets.append(offset + idx)
        values.append(int(value))

    node_colors = sankey_node_colors(len(labels))
    link_colors = sankey_link_colors(len(values))

    sankey = go.Sankey(
        node=dict(
            label=labels,
            pad=SANKEY_NODE_PAD,
            thickness=SANKEY_NODE_THICKNESS,
            color=node_colors,
            line=dict(color=SANKEY_NODE_LINE_COLOR, width=SANKEY_NODE_LINE_WIDTH),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
        arrangement="snap",
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(title=None, width=860, height=560)

    path = OUTPUT_DIR / "signal_mix_sankey.html"
    return _write_plotly_html(fig, path)


def country_fact_sheet(companies: pd.DataFrame) -> Path:
    agg = (
        companies.groupby(["country", "industry"], as_index=False)
        .agg(
            total_strength=("demand_score", "sum"),
            companies_count=("company_id", "count"),
            intensity=("intensity_score", "sum"),
        )
    )
    agg = agg[agg["intensity"] > 0].copy()
    agg = agg.sort_values(["country", "intensity"], ascending=[True, False])
    agg["industry_display"] = apply_industry_labels(agg["industry"])
    agg["industry_wrapped"] = agg["industry_display"].apply(lambda val: wrap_label(val, width=14))
    fig = px.treemap(
        agg,
        path=["country", "industry_wrapped"],
        values="intensity",
        color="total_strength",
        color_continuous_scale=[step[1] for step in MARKET_RADAR_SEQUENTIAL],
        title="Fichas por país: industrias destacadas por intensidad",
    )
    fig.update_coloraxes(colorscale=MARKET_RADAR_SEQUENTIAL, colorbar=colorbar_defaults("total_strength"))
    fig.update_traces(
        texttemplate="%{label}",
        textfont=dict(size=12, family=PRIMARY_FONT, color=PRIMARY_TEXT_COLOR),
        insidetextfont=dict(size=10, family=PRIMARY_FONT, color=PRIMARY_TEXT_COLOR),
    )
    fig.update_layout(title=None, width=880, height=680)

    path = OUTPUT_DIR / "country_fact_sheet.html"
    return _write_plotly_html(fig, path)


def main() -> None:
    events = _prep_events()
    universe = _prep_universe()
    focus_events = events[events["country"].isin(FOCUS_COUNTRIES)].copy()
    focus_universe = universe[universe["country"].isin(FOCUS_COUNTRIES)].copy()

    focus_events_universe = _filter_events_by_universe(focus_events, focus_universe)

    companies_all = _company_rollup(events)
    companies_focus = _company_rollup(focus_events)
    companies_focus_universe = _company_rollup(focus_events_universe)

    outputs = [
        coverage_summary(companies_focus_universe, focus_universe),
        total_signal_summary(companies_all),
        coverage_timeline(focus_events),
        coverage_gaps(companies_focus_universe, focus_universe),
        map_intensity(companies_all),
        ranking_companies(companies_focus),
        coverage_indicators(companies_focus),
        coverage_country_industry(focus_events),
        country_fact_sheet(companies_focus),
        signal_type_distribution(focus_events),
        signal_mix_sankey(focus_events, focus_universe),
    ]

    print("Visualizaciones generadas:")
    for path in outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
