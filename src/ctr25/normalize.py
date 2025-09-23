import pandas as pd
from pathlib import Path


COMMON_COLS = [
"company_id", "company_name", "country", "industry", "size_bin",
"source", "signal_type", "signal_strength", "ts", "url", "title", "text_snippet"
]


def combine_mock_signals():
base = [
pd.read_csv("data/samples/signals_jobs_mock.csv"),
pd.read_csv("data/samples/signals_memberships_mock.csv"),
pd.read_csv("data/samples/signals_news_mock.csv"),
]
df = pd.concat(base, ignore_index=True)[COMMON_COLS]
out = Path("data/processed/events_normalized.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Signals normalized → {out}")