"""Build exploratory HTML visuals for the LATAM climate/digital demand radar."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import numpy as np

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
    events["country"] = events["country"].fillna("")
    events["industry"] = events["industry"].fillna("")
    events["signal_type"] = events["signal_type"].fillna("unknown")
    events["source"] = events["source"].fillna("unknown")
    return events


def _company_rollup(events: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events.groupby(
            [
                "company_id",
                "company_qid",
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
    fig.write_html(path, include_plotlyjs="cdn")
    return path


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
    fig.write_html(path, include_plotlyjs="cdn")
    return path


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
    path = OUTPUT_DIR / "map_intensity.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


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


def ranking_companies(companies: pd.DataFrame) -> Path:
    ranking = companies.sort_values("demand_score", ascending=False).head(25).copy()
    ranking["last_ts_fmt"] = ranking["last_ts"].dt.strftime("%Y-%m-%d").fillna("—")

    fig = px.bar(
        ranking,
        x="demand_score",
        y="company_name",
        orientation="h",
        color="country",
        text="total_events",
        hover_data={"last_ts_fmt": True, "industry": True, "avg_strength": ":.2f"},
        title="Top 25 empresas por intensidad de demanda",
        labels={"demand_score": "Score ponderado", "total_events": "# señales", "last_ts_fmt": "Última señal"},
    )
    fig.update_layout(height=800, width=950, yaxis=dict(automargin=True))

    # Export plain text summary
    lines = ["Top 25 empresas por intensidad de demanda\n"]
    for idx, row in ranking.iterrows():
        score = f"{row['demand_score']:.2f}"
        avg = f"{row['avg_strength']:.2f}"
        lines.append(
            f"- {row['company_name']} ({row['country']} · {row['industry']}): score {score}, promedio {avg}, señales {row['total_events']}, última {row['last_ts_fmt']}"
        )
    TOP25_PATH.write_text("\n".join(lines), encoding="utf-8")

    path = OUTPUT_DIR / "ranking_companies.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


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
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def coverage_indicators(companies: pd.DataFrame, universe: pd.DataFrame) -> Path:
    active_companies = companies["company_id"].dropna().unique().tolist()
    universe = universe.copy()
    universe["company_id"] = pd.to_numeric(universe["company_id"], errors="coerce")
    universe["active"] = universe["company_id"].isin(active_companies)

    by_industry = (
        universe.groupby("industry", as_index=False)
        .agg(total=("company_id", "count"), active=("active", "sum"))
    )
    by_industry["coverage_pct"] = (by_industry["active"] / by_industry["total"]).fillna(0) * 100
    by_industry = by_industry.sort_values("coverage_pct", ascending=False)

    fig = px.bar(
        by_industry,
        x="industry",
        y="coverage_pct",
        text="active",
        title="Cobertura (%) de empresas con señales por industria",
        labels={"coverage_pct": "% empresas activas"},
    )
    fig.update_layout(width=950, height=500, xaxis=dict(automargin=True))

    path = OUTPUT_DIR / "coverage_indicators.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


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
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def main() -> None:
    events = _prep_events()
    universe = _prep_universe()
    keywords = _load_keywords(KEYWORDS_PATH)
    companies = _company_rollup(events)

    outputs = [
        heatmap_country_industry(companies),
        timeline_signals(events),
        map_intensity(companies),
        ranking_companies(companies),
        sentiment_distribution(companies),
        coverage_indicators(companies, universe),
        country_fact_sheet(companies),
    ]
    outputs.extend(network_themes(events, keywords))

    print("Visualizaciones generadas:")
    for path in outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
