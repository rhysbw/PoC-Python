# sportstats_app.py
"""
Streamlit GUI for SPORTSTATS‑Lite  ⭐️⚽️📈
==================================================
A drag‑and‑drop front‑end layered on top of **poc_sportstats.py**.

------------------------------------------------------------------
**When will you need to modify this file?**
------------------------------------------------------------------
This UI calls functions and constants imported from `poc_sportstats.py`.
Only change *this* file if *any* of the items below change in the back‑end:
1.  **Module name** – If you rename or move `poc_sportstats.py`, update the
    `import ... from poc_sportstats` lines near the top.
2.  **Public symbols** – The GUI relies on:
      - `DEFAULT_MAPPING`
      - `ColumnMapper`
      - `filter_rows`
      - `compute_counts`
      - `two_prop_z`
      - `diff_ci`
      - `narrative`
    If you rename, remove, or change the signatures/return types of any of
    these, adjust the corresponding calls below.
3.  **Mapping keys** – If the canonical keys (`team`, `half`, `pass_type`,
    `outcome`) change, update the loop that builds sidebar mapping inputs.
4.  **Narrative output** – The UI outputs the text from `narrative()`
    verbatim as Markdown.  If `narrative` changes to return plain text or a
    different format, tweak the `st.markdown()` call.

Everything else (layout, colours, styling) lives entirely in this file and
can be modified without touching the CLI back‑end.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

# ---- Import PUBLIC API from the CLI script ----
from poc_sportstats import (
    DEFAULT_MAPPING,
    ColumnMapper,
    filter_rows,
    compute_counts,
    two_prop_z,
    diff_ci,
    narrative,
)

# ------------------------------------------------
# Page config & helper CSS
# ------------------------------------------------
st.set_page_config(
    page_title="SPORTSTATS‑Lite",
    page_icon="⚽",
    layout="wide",
)

# soft pastel background & tweaked markdown width
st.markdown(
    """
<style>
    body {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stRadio > div {
        flex-direction: row;   /* horizontal radio buttons */
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------
# Sidebar – file upload & parameters
# ------------------------------------------------

def sidebar_ui():
    st.sidebar.header("1 Upload CSV")
    csv_file = st.sidebar.file_uploader("Drag or browse a Sportscode CSV", type="csv")

    st.sidebar.header("2 Column mapping (optional)")
    mapping_inputs: dict[str, str] = {}
    with st.sidebar.expander("Show / edit column names", expanded=False):
        for key, default in DEFAULT_MAPPING.items():
            mapping_inputs[key] = st.text_input(f"{key} column", value=default, key=key)

    st.sidebar.header("3 Filters & parameters")
    team_a = st.sidebar.text_input("Team A name ✏️")
    team_b = st.sidebar.text_input("Team B name ✏️")

    st.sidebar.subheader("Pass filters")
    completed_only = st.sidebar.checkbox("Completed passes only", value=True)
    pass_type_raw = st.sidebar.text_input("Pass Type filter (comma-separated)")
    pass_types = [s.strip() for s in pass_type_raw.split(",") if s.strip()]

    st.sidebar.subheader("Statistical settings")
    alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.005)
    show_ci = st.sidebar.checkbox("Show confidence interval", value=True)

    return (
        csv_file,
        mapping_inputs,
        team_a,
        team_b,
        completed_only,
        pass_types,
        alpha,
        show_ci,
    )


# ------------------------------------------------
# Main app logic
# ------------------------------------------------

def main():
    (
        csv_file,
        mapping_inputs,
        team_a,
        team_b,
        completed_only,
        pass_types,
        alpha,
        show_ci,
    ) = sidebar_ui()

    st.title("⚽ SPORTSTATS-Lite")
    st.caption("Instant z-test insights on Hudl Sportscode exports")

    if not csv_file:
        st.info("⬅️  Upload a CSV to get started.")
        return

    # ---- Build mapper and validate columns ----
    mapper = ColumnMapper(None)
    mapper._map.update(mapping_inputs)  # type: ignore[attr-defined]

    df_raw = pd.read_csv(csv_file)
    missing_cols = {mapper[k] for k in ("team", "half", "pass_type", "outcome")} - set(
        df_raw.columns
    )
    if missing_cols:
        st.error(
            f"❌  CSV is missing required columns: {', '.join(sorted(missing_cols))}. "
            "Check your mapping names."
        )
        return

    # ---- Filters ----
    df = filter_rows(df_raw, mapper, completed_only, pass_types)
    if df.empty:
        st.warning("No rows left after applying filters.")
        return

    # ---- Ensure team names present ----
    if not (team_a and team_b):
        st.info("⬅️  Enter both team names in the sidebar to run the analysis.")
        return

    metrics = compute_counts(df, mapper)
    if {team_a, team_b} - metrics.keys():
        st.error("One or both teams not found in the filtered data.")
        return

    m1, m2 = metrics[team_a], metrics[team_b]
    z, p = two_prop_z(
        m1["passes_in_opp_half"],
        m1["total_passes"],
        m2["passes_in_opp_half"],
        m2["total_passes"],
    )

    ci = (
        diff_ci(
            m1["passes_in_opp_half"],
            m1["total_passes"],
            m2["passes_in_opp_half"],
            m2["total_passes"],
            alpha,
        )
        if show_ci
        else None
    )

    # ------------------------------------------------
    # Display: metrics & interpretation
    # ------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{team_a} passes in opp. half", value=f"{m1['passes_in_opp_half']} / {m1['total_passes']}")
    with col2:
        st.metric(label=f"{team_b} passes in opp. half", value=f"{m2['passes_in_opp_half']} / {m2['total_passes']}")

    st.subheader("Interpretation 🧠")
    st.markdown(narrative(team_a, team_b, m1, m2, z, p, alpha, ci))

    with st.expander("See filtered data", expanded=False):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
