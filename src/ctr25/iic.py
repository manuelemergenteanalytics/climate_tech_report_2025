import pandas as pd
return df.groupby(by)[col].transform(z)




def compute():
events = pd.read_csv("data/processed/events_normalized.csv", parse_dates=["ts"])
# Mock: map signals → pilares via simple rules
map_signal_to_pilar = {
"job_posting": "H",
"sbti": "C",
"re100": "C",
"green_bond": "F",
"pilot_news": "P",
"newsroom": "X",
}
events["pilar"] = events["signal_type"].map(map_signal_to_pilar).fillna("X")


# Aggregate signal_strength by company × pilar
agg = events.pivot_table(
index=["company_id", "company_name", "country", "industry", "size_bin"],
columns="pilar", values="signal_strength", aggfunc="sum", fill_value=0
).reset_index()


for p in PILARS:
if p not in agg.columns:
agg[p] = 0.0


# Winsorize p95 per pilar (simple clip)
for p in PILARS:
q95 = agg[p].quantile(0.95)
agg[p] = agg[p].clip(upper=q95)


by = ["country", "industry", "size_bin"]
for p in PILARS:
agg[f"{p}_z"] = _z_by_stratum(agg, p, by)


with open("config/weights.yml", "r", encoding="utf-8") as f:
w = yaml.safe_load(f)


agg["IIC_raw"] = (
w["pilar_weights"]["H"] * agg["H_z"] +
w["pilar_weights"]["C"] * agg["C_z"] +
w["pilar_weights"]["F"] * agg["F_z"] +
w["pilar_weights"]["P"] * agg["P_z"] +
w["pilar_weights"]["X"] * agg["X_z"]
)


# Rebase 0–100 per stratum
def rebase(x):
mn, mx = x.min(), x.max()
if mx - mn < 1e-9:
return np.full_like(x, 50.0)
return 100 * (x - mn) / (mx - mn)


agg["IIC"] = agg.groupby(by)["IIC_raw"].transform(rebase)


out = Path("data/processed/company_scores.csv")
agg.to_csv(out, index=False)
print(f"IIC computed → {out}")