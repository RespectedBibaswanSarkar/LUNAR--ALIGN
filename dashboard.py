#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="LUNAR-ALIGN Dashboard", page_icon="🌙", layout="wide")
st.title("🌙 LUNAR-ALIGN Performance Dashboard")
st.caption("Live tracking of registration quality for lunar image pairs.")


def load_summary(base_dir: Path):
    summary_path = base_dir / "batch_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    results = []
    for metrics_file in sorted(base_dir.rglob("metrics.json")):
        try:
            with open(metrics_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["pair_id"] = metrics_file.parent.name
            results.append(payload)
        except Exception:
            continue

    if not results:
        return pd.DataFrame(columns=["pair_id", "rmse", "inlier_ratio", "spatial_coverage", "runtime", "matcher"])

    df = pd.DataFrame(results)
    df = df.rename(columns={"spatial_coverage": "coverage"})
    return df[["pair_id", "rmse", "inlier_ratio", "coverage", "runtime", "matcher"]]


base_dir = Path("outputs")
summary = load_summary(base_dir)

if summary.empty:
    st.warning("No results found yet. Run the dataset pipeline first to populate outputs/ with metrics.")
st.stop()

summary = summary.sort_values("inlier_ratio", ascending=False).reset_index(drop=True)

metrics = {
    "Mean RMSE": float(summary["rmse"].mean()),
    "Mean Inlier Ratio": float(summary["inlier_ratio"].mean()),
    "Mean Coverage": float(summary["coverage"].mean()),
    "Mean Runtime (s)": float(summary["runtime"].mean()),
}

cols = st.columns(4)
for i, (label, value) in enumerate(metrics.items()):
    cols[i].metric(label, f"{value:.4f}")

st.subheader("Pair-by-pair quality table")
filtered = summary[["pair_id", "matcher", "rmse", "inlier_ratio", "coverage", "runtime"]]
st.dataframe(filtered, use_container_width=True)

st.subheader("Visual diagnostics")
chart_col, dist_col = st.columns(2)
with chart_col:
    st.line_chart(summary.set_index("pair_id")["inlier_ratio"], use_container_width=True)
with dist_col:
    st.bar_chart(summary.set_index("pair_id")["rmse"], use_container_width=True)

st.subheader("Quick interpretation")
if "inlier_ratio" in summary.columns:
    best = summary.loc[summary["inlier_ratio"].idxmax()]
    st.write(f"Best-performing pair: {best['pair_id']} with inlier ratio {best['inlier_ratio']:.4f} and RMSE {best['rmse']:.6f}.")
