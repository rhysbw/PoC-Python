"""
Streamlit GUI for SPORTSTATS‑Lite (generic)
-------------------------------------------
Drag‑and‑drop your CSV, choose:

* Group column + the two groups to compare
* Metric column + which values count as “success”
* Any number of include‑filters
* Significance level & CI

Runs entirely on top of `sportstats_logic.py`.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from sportstats_logic import (
    ColumnMapper,
    DEFAULT_MAPPING,
    apply_filters,
    compute_binary_metric,
    diff_ci,
    narrative,
    parse_filter_string,
    two_prop_z,
)

st.set_page_config(page_title="SPORTSTATS‑Lite", page_icon="⚽", layout="wide")

st.title("⚽ SPORTSTATS‑Lite – Generic z‑test explorer")
st.caption("Upload a Sportscode CSV (or any similar dataset) and compare any binary metric between two groups.")

# ------------------ Upload ------------------
csv_file = st.file_uploader("📄 Upload CSV", type="csv")

if csv_file is None:
    st.info("Awaiting CSV…")
    st.stop()

df_raw = pd.read_csv(csv_file)

# ------------------ Mapping pane ------------
with st.expander("Column mapping (optional)"):
    mapping_inputs: dict[str, str] = {}
    for key, default in DEFAULT_MAPPING.items():
        mapping_inputs[key] = st.text_input(f"{key} column", value=default)
mapper = ColumnMapper(None)
mapper._map.update(mapping_inputs)  # type: ignore[attr-defined]

# ------------------ Group selection ---------
group_col = st.selectbox("Grouping column (teams, categories…)", df_raw.columns, index=0)
unique_groups = sorted(df_raw[group_col].dropna().astype(str).unique())
col1, col2 = st.columns(2)
group_a = col1.selectbox("Group A", unique_groups, index=0)
group_b_options = [g for g in unique_groups if g != group_a]
group_b = col2.selectbox("Group B", group_b_options, index=0)

# ------------------ Metric selection --------
metric_col = st.selectbox("Metric column (binary success definition)", df_raw.columns, index=0)
metric_vals = sorted(df_raw[metric_col].dropna().astype(str).unique())
success_vals = st.multiselect("Values that count as success", metric_vals)

if not success_vals:
    st.warning("Choose at least one success value to continue.")
    st.stop()

# ------------------ Filters -----------------
st.subheader("Optional include‑filters")
filter_rules = []
for i in range(3):  # allow up to 3 quick filters
    cols = st.selectbox(f"Filter {i+1} column", ["(none)"] + list(df_raw.columns), key=f"fcol{i}")
    if cols != "(none)":
        vals = sorted(df_raw[cols].dropna().astype(str).unique())
        chosen = st.multiselect(f"Keep rows where **{cols}** is:", vals, key=f"fval{i}")
        if chosen:
            filter_rules.append((cols, chosen))

alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.005)
show_ci = st.checkbox("Show confidence interval", value=True)

# ------------------ Run analysis ------------
df_filt = apply_filters(df_raw, filter_rules)

if df_filt.empty:
    st.error("No rows left after filtering — adjust your filters.")
    st.stop()

metrics = compute_binary_metric(df_filt, group_col, metric_col, success_vals)

if {group_a, group_b} - metrics.keys():
    st.error("One or both groups not present after filtering.")
    st.stop()

m1, m2 = metrics[group_a], metrics[group_b]
z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if show_ci else None

# ------------------ Display -----------------
st.header("Results")
left, right = st.columns(2)
left.metric(f"{group_a} successes", f"{m1['success']} / {m1['total']}")
right.metric(f"{group_b} successes", f"{m2['success']} / {m2['total']}")

st.markdown(
    narrative(
        group_a,
        group_b,
        m1,
        m2,
        z,
        p,
        alpha,
        ci,
        metric_desc=f'“{metric_col} ∈ {success_vals}”',
    )
)

with st.expander("Show filtered data"):
    st.dataframe(df_filt, use_container_width=True)
