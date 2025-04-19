"""
SPORTSTATS-Lite – Streamlit GUI with Insight-Suggestor
======================================================
• Upload CSV → optional AI-free suggestions (heuristics)
• Config-file mode OR manual selections
• Two-proportion or lag-sequential tests with narrative
"""

from __future__ import annotations
import json, yaml
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from sportstats_logic import (
    apply_filters, compute_binary_metric, compute_lag_sequential,
    diff_ci, prop_narrative, parse_filter_string, two_prop_z, lag_narrative,
    suggest_hypotheses
)

st.set_page_config(page_title="SPORTSTATS-Lite Wizard", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
        .stAppDeployButton {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)
st.title("⚽ SPORTSTATS-Lite")

# -------------- 1. Upload CSV ------------------------------------------
csv_buf = st.file_uploader("📄 Upload a Sportscode CSV", type="csv")
if not csv_buf:
    st.info("Please upload a CSV to begin.")
    st.stop()
df_raw = pd.read_csv(csv_buf)

# -------------- 2. Suggest insights ------------------------------------
with st.expander("🔎 Suggested insights (click to auto-fill)"):
    suggestions = suggest_hypotheses(df_raw)
    sugg_labels = ["(none)"] + [s["readable"] for s in suggestions]
    choice = st.radio("Select a suggestion", sugg_labels, index=0, key="sugg_choice")

    if choice != "(none)":
        picked = next(s for s in suggestions if s["readable"] == choice)
        st.success("Suggestion loaded into sidebar — feel free to tweak.")
        # Stash picked dict into session_state for later use
        st.session_state["picked_suggestion"] = picked
    else:
        st.session_state.pop("picked_suggestion", None)

# -------------- 3. Mode: manual or config ------------------------------
mode = st.radio("Mode", ["Manual input", "Config file"])
cfg: dict = {}
if mode == "Config file":
    cfg_up = st.file_uploader("YAML/JSON config")
    if cfg_up:
        text = cfg_up.read().decode("utf-8")
        cfg = yaml.safe_load(text) if cfg_up.name.lower().endswith((".yml", ".yaml")) else json.loads(text)
    else:
        st.stop()

# -------------- 4. Resolve parameters ----------------------------------
def get_from_suggestion(field, default=None):
    sugg = st.session_state.get("picked_suggestion", {})
    return sugg.get(field, default)

if mode == "Manual input":
    st.header("Define parameters")
    group_col = st.selectbox(
        "Group column", df_raw.columns,
        index=list(df_raw.columns).index(get_from_suggestion("group_column", df_raw.columns[0]))
    )
    groups = sorted(df_raw[group_col].dropna().astype(str).unique())
    default_a, default_b = get_from_suggestion("groups", groups[:2])
    group_a = st.selectbox("Group A", groups, index=groups.index(default_a) if default_a in groups else 0)
    group_b = st.selectbox("Group B", [g for g in groups if g != group_a],
                           index=[g for g in groups if g != group_a].index(default_b) if default_b in groups else 0)

    metric_col = st.selectbox(
        "Metric column", df_raw.columns,
        index=list(df_raw.columns).index(get_from_suggestion("metric_column", df_raw.columns[0]))
    )
    vals = sorted(df_raw[metric_col].dropna().astype(str).unique())
    success_vals = st.multiselect(
        "Success values", vals,
        default=get_from_suggestion("success_vals", [])
    )

    filters: list[tuple[str, list[str]]] = []
    with st.expander("Filters"):
        fcol = st.selectbox("Filter column", ["(none)"] + list(df_raw.columns))
        if fcol != "(none)":
            chosen = st.multiselect("Keep rows where value is:", sorted(df_raw[fcol].unique()))
            if chosen:
                filters.append((fcol, chosen))

    alpha = st.slider("α", 0.01, 0.10, float(get_from_suggestion("alpha", 0.05)), 0.005)
    test_type_default = "Lag-sequential" if get_from_suggestion("test") == "lag" else "Proportion"
    test_type = st.radio("Test type", ["Proportion", "Lag-sequential"], index=0 if test_type_default=="Proportion" else 1)
    show_ci = st.checkbox("Show CI (proportion only)", True)
else:
    group_col = cfg["group_column"]; group_a, group_b = cfg["groups"]
    metric_col = cfg["metric_column"]; success_vals = cfg["metric_success"].split(",")
    filters = [parse_filter_string(f) for f in cfg.get("filters", [])]
    alpha = cfg.get("alpha", 0.05)
    test_type = "Lag-sequential" if cfg.get("test") == "lag" else "Proportion"
    show_ci = cfg.get("ci", True)

if not success_vals:
    st.warning("Pick at least one success value.")
    st.stop()

df_filt = apply_filters(df_raw, filters)
if df_filt.empty:
    st.error("No rows left after filters.")
    st.stop()

# -------------- 5. Run analysis ----------------------------------------
st.header("Results")
if test_type == "Proportion":
    met = compute_binary_metric(df_filt, group_col, metric_col, success_vals)
    m1, m2 = met[group_a], met[group_b]
    z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
    ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if show_ci else None
    st.metric(f"{group_a} successes", f"{m1['success']} / {m1['total']}")
    st.metric(f"{group_b} successes", f"{m2['success']} / {m2['total']}")
    st.markdown(prop_narrative(group_a, group_b, m1, m2, z, p, alpha, ci,
                          metric_desc=f"{metric_col} ∈ {success_vals}"))
else:
    stats = compute_lag_sequential(df_filt, group_col, metric_col, success_vals)
    st.markdown(lag_narrative(group_a, stats[group_a], alpha))
    st.markdown(lag_narrative(group_b, stats[group_b], alpha))

with st.expander("Filtered data"):
    st.dataframe(df_filt, use_container_width=True)
