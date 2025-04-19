"""
Streamlit GUI for SPORTSTATS‑Lite (generic, config‑aware)
=========================================================
Choose between:
  • *Config mode* – upload a YAML/JSON config (same schema as CLI)
  • *Manual mode* – pick group/metric/filters interactively
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from sportstats_logic import (
    apply_filters,
    compute_binary_metric,
    diff_ci,
    load_config,
    narrative,
    parse_filter_string,
    two_prop_z,
)

st.set_page_config(page_title="SPORTSTATS‑Lite", page_icon="⚽", layout="wide")
st.title("⚽ SPORTSTATS‑Lite")

# ------------------ Upload CSV ------------------
csv_file = st.file_uploader("📄 Upload Sportscode CSV", type="csv")
if csv_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df_raw = pd.read_csv(csv_file)

# ------------------ Choose mode -----------------
mode = st.radio("Analysis mode", ["Manual", "Config file"])
cfg: dict = {}

if mode == "Config file":
    cfg_file = st.file_uploader("Upload YAML/JSON config", type=["yml", "yaml", "json"])
    if cfg_file:
        cfg = load_config(Path(cfg_file.name)).copy()  # type: ignore[arg-type]
        st.success("Config loaded – settings locked.")
    else:
        st.warning("Upload a config file or switch to Manual mode.")
        st.stop()

# ------------------ Resolve parameters ----------
if mode == "Manual":
    group_col = st.selectbox("Grouping column. What we want to group py (Example: Your teams column)", df_raw.columns, index=0)
    unique_groups = sorted(df_raw[group_col].dropna().astype(str).unique())
    col1, col2 = st.columns(2)
    group_a = col1.selectbox("Group A", unique_groups, index=0)
    group_b = col2.selectbox("Group B", [g for g in unique_groups if g != group_a], index=0)

    metric_col = st.selectbox("Metric column. What we want to mesure (Example: Your Half Column)", df_raw.columns, index=0)
    metric_vals = sorted(df_raw[metric_col].dropna().astype(str).unique())
    success_vals = st.multiselect("Values that count as success (Example: Opposition Half)", metric_vals)

    filter_rules = []
    with st.expander("Optional include‑filters (Example: Pass Type and only include 'Regular Pass')"):
        for i in range(3):
            cols = st.selectbox(f"Filter {i+1} column", ["(none)"] + list(df_raw.columns), key=f"fcol{i}")
            if cols != "(none)":
                vals = sorted(df_raw[cols].dropna().astype(str).unique())
                chosen = st.multiselect(f"{cols} ∈", vals, key=f"fval{i}")
                if chosen:
                    filter_rules.append((cols, chosen))

    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.005)
    show_ci = st.checkbox("Show confidence interval", value=True)
else:  # Config mode
    try:
        group_col = cfg["group_column"]
        group_a, group_b = cfg["groups"]
        metric_col = cfg["metric_column"]
        success_vals = cfg["metric_success"].split(",") if isinstance(cfg["metric_success"], str) else list(cfg["metric_success"])
        filter_rules = [parse_filter_string(f) for f in cfg.get("filters", [])]
        alpha = cfg.get("alpha", 0.05)
        show_ci = cfg.get("ci", True)
    except KeyError as err:
        st.error(f"Config missing key: {err}")
        st.stop()

# ------------------ Validate manual selections --
if mode == "Manual" and not success_vals:
    st.warning("Choose at least one success value.")
    st.stop()

# ------------------ Apply filters ----------------
df_filt = apply_filters(df_raw, filter_rules)

if df_filt.empty:
    st.error("No rows left after filtering.")
    st.stop()

metrics = compute_binary_metric(df_filt, group_col, metric_col, success_vals)
if {group_a, group_b} - metrics.keys():
    st.error("One or both groups missing after filtering.")
    st.stop()

m1, m2 = metrics[group_a], metrics[group_b]
z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if show_ci else None

# ------------------ Display results ---------------
st.header("Results")
left, right = st.columns(2)
left.metric(f"{group_a} successes", f"{m1['success']} / {m1['total']}")
right.metric(f"{group_b} successes", f"{m2['success']} / {m2['total']}")

metric_desc = f'“{metric_col} ∈ {success_vals}”'
st.markdown(narrative(group_a, group_b, m1, m2, z, p, alpha, ci, metric_desc=metric_desc))

with st.expander("Filtered data"):
    st.dataframe(df_filt, use_container_width=True)
