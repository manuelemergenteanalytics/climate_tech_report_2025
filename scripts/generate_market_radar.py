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


def heatmap_country_industry(companies: pd.DataFrame) -> Path:
    pivot = (
        companies.groupby(["country", "industry"], as_index=False)
        .agg(total_score=("demand_score", "sum"), companies=("company_id", "count"))
    )
    if pivot.empty:
        pivot = pd.DataFrame({"country": [], "industry": [], "total_score": []})

    fig = px.density_heatmap(
        pivot,
        x="industry",
        y="country",
        z="total_score",
        color_continuous_scale="Tealrose",
        title="Intensidad de señales por país e industria",
        labels={"total_score": "Σ score empresas"},
    )
    fig.update_layout(width=900, height=600, yaxis=dict(categoryorder="total ascending"))

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

    fig = px.bar(
        timeline,
        x="month",
        y="strength",
        color="signal_type",
        barmode="group",
        title="Cronología de señales por tipo",
        labels={"strength": "Σ score empresas", "month": "Mes"},
    )
    fig.update_layout(width=1000, height=500, bargap=0.15)

    path = OUTPUT_DIR / "timeline_signals.html"
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
        color_continuous_scale="Viridis",
    )
    fig.update_layout(width=900, height=550)
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=-15, lon=-70),
        lataxis=dict(range=[-60, 30]),
        lonaxis=dict(range=[-120, -30]),
        showcountries=True,
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

    html_lines = [
        "<html>",
        "<head>",
        "\t<meta charset=\"utf-8\" />",
        "\t<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>",
        f"\t{PLOTLYJS_SCRIPT}",
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
    ]

    path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    return path


def ranking_companies(companies: pd.DataFrame) -> Path:
    ranking = (
        companies.sort_values("demand_score", ascending=False)
        .head(25)
        .copy()
    )
    ranking["last_ts_fmt"] = ranking["last_ts"].dt.strftime("%Y-%m-%d").fillna("—")

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
    )
    fig.update_layout(height=800, width=950, yaxis=dict(automargin=True))

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

    fig = px.bar(
        rollup,
        x="total_demand",
        y="industry",
        orientation="h",
        text="total_companies",
        title="Industrias con mayor intensidad de señales",
        labels={
            "total_demand": "Σ score ponderado",
            "industry": "Industria",
            "total_companies": "Empresas",
        },
        hover_data={
            "total_events": True,
            "median_intensity": ":.2f",
            "avg_events_per_company": True,
        },
    )
    fig.update_layout(width=950, height=550, yaxis=dict(automargin=True))

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

    if coverage.empty:
        coverage = pd.DataFrame({"country": [], "industry": [], "total_events": []})

    fig = px.density_heatmap(
        coverage,
        x="industry",
        y="country",
        z="total_events",
        color_continuous_scale="Tealgrn",
        title="Señales por país e industria",
        labels={"total_events": "# señales"},
    )

    if not coverage.empty:
        custom = coverage[["companies", "total_demand"]].to_numpy()
        fig.update_traces(
            customdata=custom,
            hovertemplate=(
                "industria=%{x}<br>país=%{y}<br>señales=%{z:.0f}<br>"
                "empresas únicas=%{customdata[0]:.0f}<br>Σ score=%{customdata[1]:.2f}<extra></extra>"
            ),
        )

    fig.update_layout(width=950, height=600, xaxis=dict(automargin=True))
    fig.update_coloraxes(colorbar=dict(title="# señales"))

    path = OUTPUT_DIR / "coverage_country_industry.html"
    return _write_plotly_html(fig, path)


def signal_type_distribution(events: pd.DataFrame) -> Path:
    counts = (
        events.groupby("signal_type", dropna=False)
        .size()
        .reset_index(name="total_signals")
    )
    counts["signal_type"] = counts["signal_type"].fillna("sin dato").astype(str)
    counts = counts.sort_values("total_signals", ascending=False)

    fig = px.bar(
        counts,
        x="signal_type",
        y="total_signals",
        text="total_signals",
        title="Distribución de señales por tipo",
        labels={"signal_type": "Tipo de señal", "total_signals": "Cantidad"},
    )
    fig.update_layout(width=750, height=450, xaxis=dict(categoryorder="total descending"))

    path = OUTPUT_DIR / "signal_type_distribution.html"
    return _write_plotly_html(fig, path)


def coverage_summary(companies: pd.DataFrame, universe: pd.DataFrame) -> Path:
    universe_ids = pd.to_numeric(universe["company_id"], errors="coerce")
    total_universe = int(universe_ids.dropna().nunique())

    events_per_company = companies["total_events"].dropna().to_numpy()
    active_ids = pd.to_numeric(companies["company_numeric_id"], errors="coerce")
    active_count = int(active_ids.dropna().nunique())
    inactive_count = max(total_universe - active_count, 0)
    coverage_pct = (active_count / total_universe * 100) if total_universe else 0.0

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

    metrics = [
        "Empresas en el universo",
        "Empresas con señales",
        "Empresas sin señales",
        "% universo cubierto",
        "Total de señales registradas",
        "Señales por empresa (media)",
        "Señales por empresa (mediana)",
        "Señales por empresa (desvío estándar)",
        "Señales por empresa (Q1)",
        "Señales por empresa (Q3)",
        "Señales por empresa (mínimo)",
        "Señales por empresa (máximo)",
    ]

    total_signals = int(events_per_company.sum()) if events_per_company.size else 0
    values = [
        f"{total_universe:,}",
        f"{active_count:,}",
        f"{inactive_count:,}",
        f"{coverage_pct:.1f}%",
        f"{total_signals:,}",
        f"{mean_signals:.2f}",
        f"{median_signals:.2f}",
        f"{std_signals:.2f}",
        f"{q1:.2f}",
        f"{q3:.2f}",
        f"{min_signals:,}",
        f"{max_signals:,}",
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Métrica", "Valor"], align="left"),
                cells=dict(values=[metrics, values], align="left"),
            )
        ]
    )
    fig.update_layout(title="Resumen estadístico de cobertura", width=700, height=480)

    path = OUTPUT_DIR / "coverage_summary.html"
    return _write_plotly_html(fig, path)


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

    sankey = go.Sankey(
        node=dict(label=labels, pad=18, thickness=18),
        link=dict(source=sources, target=targets, value=values),
        arrangement="snap",
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title="Flujo de cobertura y mezcla de señales",
        width=900,
        height=600,
    )

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
    fig = px.treemap(
        agg,
        path=["country", "industry"],
        values="intensity",
        color="total_strength",
        color_continuous_scale="Bluered",
        title="Fichas por país: industrias destacadas por intensidad",
    )
    fig.update_layout(width=950, height=600)

    path = OUTPUT_DIR / "country_fact_sheet.html"
    return _write_plotly_html(fig, path)


def main() -> None:
    events = _prep_events()
    universe = _prep_universe()
    focus_events = events[events["country"].isin(FOCUS_COUNTRIES)].copy()
    focus_universe = universe[universe["country"].isin(FOCUS_COUNTRIES)].copy()

    companies_all = _company_rollup(events)
    companies_focus = _company_rollup(focus_events)

    outputs = [
        coverage_summary(companies_focus, focus_universe),
        heatmap_country_industry(companies_focus),
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
