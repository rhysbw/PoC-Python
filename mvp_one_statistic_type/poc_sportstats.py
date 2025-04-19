#!/usr/bin/env python3
"""
SPORTSTATS-Lite – PoC v3
--------------------------------------------------
* Import Sportscode CSV     (pandas)
* Optional YAML/JSON mapper for column names
* Filter rows (completed-only, pass types)
* Two-proportion z-test     (+ 95 % CI if requested)
* User-friendly summary
* Basic validation & logging

Example
-------
python poc_sportstats.py \
    --csv match.csv \
    --team-a "Barcelona" \
    --team-b "Deportivo Alavés" \
    --completed-only \
    --pass-type "Regular Pass" \
    --map mapping.yml \
    --ci
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None  # YAML support becomes optional


# ----------------------------------------------------------------------
# 1.  Column-mapping helper
# ----------------------------------------------------------------------
DEFAULT_MAPPING: dict[str, str] = {
    "team": "Row",
    "half": "Half",
    "pass_type": "Pass Type",
    "outcome": "Outcome",
}


class ColumnMapper:
    """
    Maps canonical field names → actual CSV column labels.

    If a mapping file is provided (YAML or JSON), it overrides defaults.
    Keys expected in the file: team, half, pass_type, outcome.
    """

    def __init__(self, mapping_path: Path | None):
        mapping: dict[str, str] = DEFAULT_MAPPING.copy()

        if mapping_path:
            try:
                if mapping_path.suffix.lower() in {".yml", ".yaml"}:
                    if yaml is None:
                        raise RuntimeError("pyyaml not installed.")
                    mapping.update(yaml.safe_load(mapping_path.read_text()))
                else:  # assume JSON
                    mapping.update(json.loads(mapping_path.read_text()))
            except Exception as exc:
                logging.error("Failed to read mapping file %s: %s", mapping_path, exc)
                sys.exit(1)

        # quick sanity check
        missing = {"team", "half", "pass_type", "outcome"} - mapping.keys()
        if missing:
            logging.error("Mapping file missing keys: %s", ", ".join(sorted(missing)))
            sys.exit(1)

        self._map = mapping

    def __getitem__(self, key: str) -> str:
        return self._map[key]


# ----------------------------------------------------------------------
# 2.  Data I/O + filtering
# ----------------------------------------------------------------------
def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read Sportscode CSV, auto-detect UTF-8/16."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-16")


def filter_rows(
    df: pd.DataFrame,
    mapper: ColumnMapper,
    completed_only: bool,
    pass_types: list[str],
) -> pd.DataFrame:
    """Apply user-requested filters; always keep rows whose pass_type contains 'Pass'."""
    col_pt, col_out = mapper["pass_type"], mapper["outcome"]

    mask = df[col_pt].astype(str).str.contains("", na=False)

    if completed_only:
        mask &= df[col_out].eq("Complete")

    if pass_types:
        mask &= df[col_pt].isin(pass_types)

    return df[mask]


# ----------------------------------------------------------------------
# 3.  Metrics + stats
# ----------------------------------------------------------------------
def compute_counts(df: pd.DataFrame, mapper: ColumnMapper) -> dict[str, dict[str, int]]:
    col_team, col_half = mapper["team"], mapper["half"]

    counts = (
        df.groupby(col_team)
        .agg(
            total_passes=(col_team, "size"),
            passes_in_opp_half=(col_half, lambda s: (s == "Opposition Half").sum()),
        )
        .to_dict("index")
    )
    return counts


def two_prop_z(success1: int, n1: int, success2: int, n2: int) -> tuple[float, float]:
    """z-statistic & two-tailed p (guarding against zero variance)."""
    p1, p2 = success1 / n1, success2 / n2
    pooled = (success1 + success2) / (n1 + n2)
    se2 = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if se2 == 0:
        return 0.0, 1.0
    z = (p1 - p2) / math.sqrt(se2)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def diff_ci(
    success1: int, n1: int, success2: int, n2: int, alpha: float
) -> tuple[float, float]:
    """Wald 1-alpha CI for the difference in proportions."""
    from scipy.stats import norm

    p1, p2 = success1 / n1, success2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = norm.ppf(1 - alpha / 2)
    return diff - z_crit * se, diff + z_crit * se


# ----------------------------------------------------------------------
# 4.  Interpretation helper
# ----------------------------------------------------------------------
def narrative(
    team_a: str,
    team_b: str,
    m1: dict[str, Any],
    m2: dict[str, Any],
    z: float,
    p: float,
    alpha: float,
    ci: tuple[float, float] | None,
) -> str:
    """Generate a user-friendly summary of the results."""
    p1 = m1["passes_in_opp_half"] / m1["total_passes"]
    p2 = m2["passes_in_opp_half"] / m2["total_passes"]
    diff_pct = (p1 - p2) * 100
    print(f'Team {team_a} = {p1:.3f} with {m1["passes_in_opp_half"]} passes in opposition half with total of {m1["total_passes"]} passes')
    print(f'Team {team_b} = {p2:.3f} with {m2["passes_in_opp_half"]} passes in opposition half with total of {m2["total_passes"]} passes')

    if diff_pct > 0:
        direction = "more"
        qty = diff_pct
    else:
        direction = "fewer"
        qty = abs(diff_pct)

    sig = "statistically *SIGNIFICANT*" if p < alpha else "not statistically significant"

    msg = (
        f"{team_a} completed {qty:.1f} percentage‑points {direction} of their passes "
        f"in the opposition half than {team_b}. This difference is {sig} "
        f"(z = {z:.2f}, p = {p:.3f})."
    )

    if ci is not None:
        lo, hi = [x * 100 for x in ci]
        msg += (
            f"\nThe {100*(1-alpha):.0f}% confidence interval for the gap is "
            f"[{lo:+.1f} pp, {hi:+.1f} pp]."
        )

    if min(m1["total_passes"], m2["total_passes"]) < 30:
        msg += "\n ⚠️  Sample sizes are small; interpret with caution."

    return msg



# ----------------------------------------------------------------------
# 5.  CLI & main
# ----------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sportstats-lite",
        description="Two-proportion z-test on Sportscode CSV exports.",
    )
    p.add_argument("--csv", required=True, type=Path, help="Path to Sportscode CSV file")
    p.add_argument("--team-a", required=True, help="Team A exact string (column 'Row')")
    p.add_argument("--team-b", required=True, help="Team B exact string (column 'Row')")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    p.add_argument("--completed-only", action="store_true", help="Keep only completed passes")
    p.add_argument(
        "--pass-type",
        action="append",
        metavar="TYPE",
        default=[],
        help="Include only these Pass Type values (repeatable).",
    )
    p.add_argument(
        "--map",
        type=Path,
        help="YAML/JSON column-mapping file (see docs).",
    )
    p.add_argument("--ci", action="store_true", help="Show confidence interval for the gap")
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default WARNING)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    mapper = ColumnMapper(args.map)

    df_raw = load_csv(args.csv)
    required_cols = {mapper[k] for k in ("team", "half", "pass_type", "outcome")}
    if missing := required_cols - set(df_raw.columns):
        logging.error("CSV is missing required columns: %s", ", ".join(sorted(missing)))
        sys.exit(1)

    df = filter_rows(df_raw, mapper, args.completed_only, args.pass_type)

    metrics = compute_counts(df, mapper)
    if {args.team_a, args.team_b} - metrics.keys():
        logging.error("One or both teams not found after filtering.")
        sys.exit(1)

    m1, m2 = metrics[args.team_a], metrics[args.team_b]
    z, p = two_prop_z(
        m1["passes_in_opp_half"], m1["total_passes"], m2["passes_in_opp_half"], m2["total_passes"]
    )

    ci = (
        diff_ci(
            m1["passes_in_opp_half"],
            m1["total_passes"],
            m2["passes_in_opp_half"],
            m2["total_passes"],
            args.alpha,
        )
        if args.ci
        else None
    )

    print("\n=== SPORTSTATS-Lite result ===")
    print(narrative(args.team_a, args.team_b, m1, m2, z, p, args.alpha, ci))


if __name__ == "__main__":
    main()
