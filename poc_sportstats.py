#!/usr/bin/env python3
"""
SPORTSTATS-Lite - PoC v2
--------------------------------------------------
Reads a Hudl Sportscode CSV, filters the data, compares the proportion of
passes in the opposition half for two teams via a two-proportion z-test, and
prints a user-friendly interpretation.

Example
-------
python poc_sportstats.py \
    --csv demo.csv \
    --team-a "Barcelona" \
    --team-b "Deportivo Alavés" \
    --completed-only \
    --pass-type "Regular Pass" \
    --alpha 0.05
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Data I/O and filtering
# ----------------------------------------------------------------------
def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load Sportscode CSV into a DataFrame (assumes first row is headers)."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:  # macOS often exports in UTF-16
        df = pd.read_csv(csv_path, encoding="utf-16")
    return df


def apply_filters(df: pd.DataFrame, completed_only: bool, pass_types: list[str]) -> pd.DataFrame:
    """Return a filtered DataFrame according to CLI options."""
    mask = pd.Series(True, index=df.index)

    # Keep only rows whose Pass Type contains 'Pass' (the default behaviour).
    mask &= df["Pass Type"].str.contains("Pass", na=False)

    if completed_only:
        mask &= df["Outcome"].eq("Complete")

    if pass_types:
        mask &= df["Pass Type"].isin(pass_types)

    return df[mask]


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame, team_col="Row", half_col="Half") -> dict:
    """
    Parameters
    ----------
    df : filtered DataFrame
    team_col : column that denotes the team
    half_col : column with 'Opposition Half' / 'Own Half'

    Returns
    -------
    dict mapping team → {'total_passes': int, 'passes_in_opponent_half': int}
    """
    counts = (
        df.groupby(team_col)
        .agg(
            total_passes=(team_col, "size"),
            passes_in_opponent_half=(half_col, lambda s: (s == "Opposition Half").sum()),
        )
        .to_dict("index")
    )
    return counts


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def two_proportion_z(success1: int, n1: int, success2: int, n2: int) -> tuple[float, float]:
    """Return z-statistic and two-tailed p value, guarding against 0 variance."""
    p1, p2 = success1 / n1, success2 / n2
    pooled = (success1 + success2) / (n1 + n2)
    se_term = pooled * (1 - pooled) * (1 / n1 + 1 / n2)

    if se_term == 0:
        return 0.0, 1.0  # no variability → no evidence of difference

    z = (p1 - p2) / math.sqrt(se_term)
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p = 2 * (1 - cdf)
    return z, p


def interpret(z: float, p: float, diff: float, alpha: float, n1: int, n2: int) -> str:
    """Build a richer plain-language explanation."""
    sig_text = (
        "a statistically **significant** difference" if p < alpha else "no significant difference"
    )
    diff_pct = diff * 100
    summary = (
        f"The analysis suggests {sig_text} between the two teams' proportions of passes "
        f"made in the opposition half (absolute gap ≈ {diff_pct:.1f} percentage points, "
        f"z = {z:.2f}, p = {p:.3f})."
    )

    # Light caveat for tiny samples
    if min(n1, n2) < 30:
        summary += (
            "⚠️ Note: sample sizes are small ("
            f"{n1} vs {n2} passes), so the estimate has wide uncertainty."
        )
    return summary


# ----------------------------------------------------------------------
# CLI glue
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two teams' passes in the opposition half via a z-test."
    )
    p.add_argument("--csv", required=True, type=Path, help="Path to Sportscode CSV")
    p.add_argument("--team-a", required=True, help="Exact name of Team A as in CSV 'Row'")
    p.add_argument("--team-b", required=True, help="Exact name of Team B as in CSV 'Row'")
    p.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default 0.05)"
    )
    p.add_argument(
        "--completed-only",
        action="store_true",
        help="Include only passes with Outcome == Complete",
    )
    p.add_argument(
        "--pass-type",
        action="append",
        default=[],
        metavar="TYPE",
        help="Restrict to specific pass type(s); can be repeated. "
        "If omitted, any row whose 'Pass Type' contains 'Pass' is included.",
    )
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    # --------------------------- Load & filter --------------------------
    df_raw = load_csv(args.csv)
    df = apply_filters(df_raw, args.completed_only, args.pass_type)

    # ------------------------ Compute per-team counts -------------------
    metrics = compute_metrics(df)
    if {args.team_a, args.team_b} - metrics.keys():
        missing = {args.team_a, args.team_b} - metrics.keys()
        print(f"❌  Data for team(s) {', '.join(missing)} not found after filtering.")
        sys.exit(1)

    m1, m2 = metrics[args.team_a], metrics[args.team_b]
    z, p = two_proportion_z(
        m1["passes_in_opponent_half"],
        m1["total_passes"],
        m2["passes_in_opponent_half"],
        m2["total_passes"],
    )

    # ----------------------------- Report ------------------------------
    diff = (m1["passes_in_opponent_half"] / m1["total_passes"]) - (
        m2["passes_in_opponent_half"] / m2["total_passes"]
    )

    print("\n=== SPORTSTATS-Lite PoC ===")
    print(f"CSV:      {args.csv}")
    print(f"Teams:    {args.team_a} vs {args.team_b}")
    print(f"Filters:  completed_only={args.completed_only}  pass_type={args.pass_type or 'any'}")
    print(f"Alpha:    {args.alpha}")
    print("\n*Raw counts*")
    for t, m in ((args.team_a, m1), (args.team_b, m2)):
        print(
            f"  {t:<20} passes_in_opphalf={m['passes_in_opponent_half']} "
            f"/ total={m['total_passes']}"
        )
    print("\n*Interpretation*")
    print(interpret(z, p, diff, args.alpha, m1["total_passes"], m2["total_passes"]))


if __name__ == "__main__":
    main()
