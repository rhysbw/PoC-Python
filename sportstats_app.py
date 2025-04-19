"""
Streamlit GUI for SPORTSTATS‑Lite (v0.4)
----------------------------------------
Upload CSV ▸ choose test ▸ see results.
Config files now load from the uploaded buffer (no FileNotFoundError).
"""

from __future__ import annotations
import json, yaml
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from sportstats_logic import (
    apply_filters, compute_binary_metric, compute_lag_sequential,
    diff_ci, narrative, parse_filter_string, two_prop_z, lag_narrative
)

st.set_page_config(page_title="SPORTSTATS‑Lite", page_icon="⚽", layout="wide")
st.title("⚽ SPORTSTATS‑Lite")

# ---------------- Upload data ----------------
csv_buf = st.file_uploader("CSV file", type="csv")
if not csv_buf:
    st.stop()
df_raw = pd.read_csv(csv_buf)

# ---------------- Mode selector --------------
mode = st.radio("Mode", ["Manual", "Config file"])
cfg = {}

if mode == "Config file":
    cfg_up = st.file_uploader("YAML/JSON config")
    if not cfg_up:
        st.stop()
    text = cfg_up.read().decode("utf-8")
    if cfg_up.name.lower().endswith((".yml", ".yaml")):
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)

# ------------- Parameter resolution ----------
if mode == "Manual":
    group_col = st.selectbox("Group column", df_raw.columns)
    groups = sorted(df_raw[group_col].dropna().astype(str).unique())
    group_a = st.selectbox("Group A", groups, 0)
    group_b = st.selectbox("Group B", [g for g in groups if g != group_a], 0)
    metric_col = st.selectbox("Metric column", df_raw.columns)
    vals = sorted(df_raw[metric_col].dropna().astype(str).unique())
    success_vals = st.multiselect("Success values", vals)
    filters: list[tuple[str, list[str]]] = []
    with st.expander("Filters"):
        fcol = st.selectbox("Column", ["(none)"] + list(df_raw.columns))
        if fcol != "(none)":
            fvals = st.multiselect("Keep values", sorted(df_raw[fcol].unique()))
            if fvals:
                filters.append((fcol, fvals))
    alpha = st.slider("α", 0.01, 0.10, 0.05, 0.005)
    test_type = st.radio("Test", ["Proportion", "Lag‑sequential"])
    show_ci = st.checkbox("Show CI (proportion only)", True)
else:
    group_col = cfg["group_column"]; group_a, group_b = cfg["groups"]
    metric_col = cfg["metric_column"]
    success_vals = cfg["metric_success"].split(",")
    filters = [parse_filter_string(f) for f in cfg.get("filters", [])]
    alpha = cfg.get("alpha", 0.05)
    test_type = "Lag‑sequential" if cfg.get("test") == "lag" else "Proportion"
    show_ci = cfg.get("ci", True)

if not success_vals:
    st.warning("Select success values."); st.stop()

df_filt = apply_filters(df_raw, filters)
if df_filt.empty:
    st.error("No rows after filtering."); st.stop()

# ---------------- Run analysis ---------------
if test_type == "Proportion":
    met = compute_binary_metric(df_filt, group_col, metric_col, success_vals)
    m1, m2 = met[group_a], met[group_b]
    z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
    ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if show_ci else None
    st.metric(f"{group_a} successes", f"{m1['success']} / {m1['total']}")
    st.metric(f"{group_b} successes", f"{m2['success']} / {m2['total']}")
    st.markdown(narrative(group_a, group_b, m1, m2, z, p, alpha, ci,
                          metric_desc=f"{metric_col}∈{success_vals}"))
else:
    stats = compute_lag_sequential(df_filt, group_col, metric_col, success_vals)
    fmt = lambda s: f"χ² {s['chi2']:.2f}, p {s['p']:.3f}, Q {s['yule_q']:.2f}"
    st.markdown(lag_narrative(group_a, stats[group_a], alpha))
    st.markdown(lag_narrative(group_b, stats[group_b], alpha))


with st.expander("Filtered data"):
    st.dataframe(df_filt, use_container_width=True)
