# src/ctr25/utils/qa.py
import os
import pandas as pd

def run_qa(universe_path="data/processed/universe_sample.csv", events_path="data/processed/events_normalized.csv",
           out_dir="data/interim/qa"):
    os.makedirs(out_dir, exist_ok=True)
    u = pd.read_csv(universe_path, dtype=str)
    e = pd.read_csv(events_path, dtype=str) if os.path.exists(events_path) else pd.DataFrame(columns=[
        "company_id","company_name","country","industry","size_bin","source","signal_type","signal_strength","ts","url","title","text_snippet"
    ])
    e["signal_strength"] = pd.to_numeric(e.get("signal_strength", 0), errors="coerce").fillna(0)

    # empresas con señal por tipo
    any_signal = e.groupby(["country","industry"])["company_id"].nunique().rename("companies_with_signal")
    total_companies = u.groupby(["country","industry"])["company_id"].nunique().rename("companies_total")
    by_type = e.groupby(["country","industry","signal_type"])["company_id"].nunique().rename("companies_with_type").reset_index()
    by_type_pivot = by_type.pivot_table(index=["country","industry"], columns="signal_type", values="companies_with_type", fill_value=0)

    # promedio de strength por tipo
    st = e.groupby(["country","industry","signal_type"])["signal_strength"].mean().round(3).reset_index()
    st_pivot = st.pivot_table(index=["country","industry"], columns="signal_type", values="signal_strength", fill_value=0).add_prefix("avg_")

    # merge
    df = pd.concat([total_companies, any_signal], axis=1).fillna(0).reset_index()
    df["coverage_pct"] = (100*df["companies_with_signal"]/df["companies_total"]).round(1)
    df = df.merge(by_type_pivot, on=["country","industry"], how="left").merge(st_pivot, on=["country","industry"], how="left")
    df = df.fillna(0)
    out = os.path.join(out_dir, "summary_by_stratum.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    return out
