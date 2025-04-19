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

# Page config and title
st.set_page_config(page_title="SPORTSTATS‑Lite Wizard", page_icon="⚽", layout="wide")
st.title("⚽ SPORTSTATS‑Lite Interactive Guide")

st.write("Welcome! This guided interface will help you step-by-step to upload your data, choose comparison settings, and see your results with helpful examples along the way.")

# Step 1: Data upload
st.header("Upload your data 📂")
st.info(
    "Your CSV should include: a column for group labels (e.g., 'team') and a column for the metric to compare (e.g., 'goal_scored')."
)
# Provide a sample CSV for reference
sample_df = pd.DataFrame({
    "team": ["A", "B", "A", "B"],
    "goal": [1, 0, 1, 1],
    "player": ["X", "Y", "Z", "W"]
})
csv_sample = sample_df.to_csv(index=False)
st.download_button(
    "Download sample CSV", csv_sample, file_name="sample_sportstats.csv", mime="text/csv"
)

csv_buf = st.file_uploader("Upload CSV file", type="csv", help="Choose the CSV file you want to analyze.")
if not csv_buf:
    st.stop()

df_raw = pd.read_csv(csv_buf)

# Step 2: Choose input mode
st.header("Select configuration mode ⚙️")
mode = st.radio(
    "Mode",
    ["Manual input", "Config file"],
    help="Manual input lets you pick each option step-by-step. Config file reads settings from a YAML/JSON you upload."
)

cfg: dict[str, any] = {}
if mode == "Config file":
    cfg_up = st.file_uploader("Upload YAML/JSON config", help="Use a file with keys: group_column, groups, metric_column, metric_success, filters, alpha, test, ci.")
    if not cfg_up:
        st.stop()
    text = cfg_up.read().decode("utf-8")
    cfg = yaml.safe_load(text) if cfg_up.name.lower().endswith(('.yml', '.yaml')) else json.loads(text)

# Step 3: Define parameters

if mode == "Manual input":
    st.header("Define comparison parameters 🔍")
    with st.container():
        st.subheader("Group selection")
        group_col = st.selectbox(
            "Group column", df_raw.columns,
            help="Select the column in your data that indicates group membership, e.g., 'team'."
        )
        groups = sorted(df_raw[group_col].dropna().astype(str).unique())
        group_a = st.selectbox("Group A", groups, index=0, help="First group to compare.")
        group_b = st.selectbox("Group B", [g for g in groups if g != group_a], index=0, help="Second group to compare.")

        st.subheader("Metric definition")
        metric_col = st.selectbox(
            "Metric column", df_raw.columns,
            help="Select the column holding the metric you want to compare, e.g., 'goal'."
        )
        vals = sorted(df_raw[metric_col].dropna().astype(str).unique())
        success_vals = st.multiselect(
            "Success values", vals,
            help="Choose which values count as 'success'. For example, set [1] for goals scored."
        )

        st.subheader("Optional filters")
        filters: list[tuple[str, list[str]]] = []
        with st.expander("Add filters to refine your dataset"): 
            col = st.selectbox("Filter column", ["(none)"] + list(df_raw.columns))
            if col != "(none)":
                selected = st.multiselect(
                    f"Keep only rows where {col} is...", sorted(df_raw[col].unique()),
                    help="Filter rows to only include specific values in this column."
                )
                if selected:
                    filters.append((col, selected))

        st.subheader("Statistical settings")
        alpha = st.slider(
            "Significance level (α)", 0.01, 0.10, 0.05, 0.005,
            help="Threshold for statistical significance. Lower α means stricter tests."
        )
        test_type = st.radio(
            "Test type", ["Proportion", "Lag‑sequential"],
            help="Proportion compares success rates; Lag‑sequential evaluates transitions between events."
        )
        show_ci = st.checkbox(
            "Show confidence interval (Proportion only)", True,
            help="Enable to display CI for the difference in proportions."
        )
else:
    group_col = cfg["group_column"]
    group_a, group_b = cfg["groups"]
    metric_col = cfg["metric_column"]
    success_vals = cfg["metric_success"].split(",")
    filters = [parse_filter_string(f) for f in cfg.get("filters", [])]
    alpha = cfg.get("alpha", 0.05)
    test_type = "Lag‑sequential" if cfg.get("test") == "lag" else "Proportion"
    show_ci = cfg.get("ci", True)

if not success_vals:
    st.warning("Please select at least one success value to proceed.")
    st.stop()

# Apply filters
df_filt = apply_filters(df_raw, filters)
if df_filt.empty:
    st.error("No data left after applying filters. Adjust your filters and try again.")
    st.stop()

# Step 4: Run analysis
st.header("View results 📊")
if test_type == "Proportion":
    met = compute_binary_metric(df_filt, group_col, metric_col, success_vals)
    m1, m2 = met[group_a], met[group_b]
    z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
    ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if show_ci else None
    st.metric(f"{group_a} successes", f"{m1['success']} / {m1['total']}")
    st.metric(f"{group_b} successes", f"{m2['success']} / {m2['total']}")
    st.markdown(narrative(
        group_a, group_b, m1, m2, z, p, alpha, ci,
        metric_desc=f"{metric_col} ∈ {success_vals}"
    ))
else:
    stats = compute_lag_sequential(df_filt, group_col, metric_col, success_vals)
    st.markdown(lag_narrative(group_a, stats[group_a], alpha))
    st.markdown(lag_narrative(group_b, stats[group_b], alpha))

with st.expander("See filtered data"):
    st.dataframe(df_filt, use_container_width=True)
