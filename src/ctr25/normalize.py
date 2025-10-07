import pandas as pd
from pathlib import Path
from .utils.text import load_industry_map, normalize_industry

COMMON_COLS = [
    "company_id", "company_name", "country", "industry", "size_bin",
    "source", "signal_type", "signal_strength", "ts", "url", "title", "text_snippet",
    "climate_score", "sentiment_label", "sentiment_score",
]

def combine_mock_signals():
    base = [
        pd.read_csv("data/samples/signals_jobs_mock.csv"),
        pd.read_csv("data/samples/signals_memberships_mock.csv"),
        pd.read_csv("data/samples/signals_news_mock.csv"),
    ]
    df = pd.concat(base, ignore_index=True)[COMMON_COLS]

    # Normalización de industria (alias -> slug canónico)
    imap = load_industry_map()  # lee config/industry_map.yml
    df["industry"] = df["industry"].apply(lambda x: normalize_industry(str(x), imap))

    out = Path("data/processed/events_normalized.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Signals normalized -> {out}")

