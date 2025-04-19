#!/usr/bin/env python3
"""
SPORTSTATS‑Lite / Logic + CLI
========================================================
Generic two‑proportion z‑test on any binary metric from a
Sportscode CSV (or any tabular file).

Key features
------------
• YAML/JSON config ( --config my_analysis.yml )  
• Fully overridable via individual CLI flags  
• Column mapping helper (if your CSV headers differ)  

Public API  (importable by GUIs):
    load_config
    ColumnMapper, DEFAULT_MAPPING
    parse_filter_string, apply_filters
    compute_binary_metric, two_prop_z, diff_ci, narrative
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None  # YAML support optional

# ------------------------------------------------------------------ #
# 1. Column mapping
# ------------------------------------------------------------------ #
DEFAULT_MAPPING: Dict[str, str] = {
    "team": "Row",
    "half": "Half",
    "pass_type": "Pass Type",
    "outcome": "Outcome",
}


class ColumnMapper:
    KEYS = {"team", "half", "pass_type", "outcome"}

    def __init__(self, mapping_path: Path | None):
        mapping: Dict[str, str] = DEFAULT_MAPPING.copy()

        if mapping_path:
            try:
                if mapping_path.suffix.lower() in {".yml", ".yaml"}:
                    if yaml is None:
                        raise RuntimeError("pyyaml not installed.")
                    mapping.update(yaml.safe_load(mapping_path.read_text()))
                else:
                    mapping.update(json.loads(mapping_path.read_text()))
            except Exception as exc:
                logging.error("Failed to read mapping %s: %s", mapping_path, exc)
                sys.exit(1)

        missing = self.KEYS - mapping.keys()
        if missing:
            logging.error("Mapping file missing keys: %s", ", ".join(sorted(missing)))
            sys.exit(1)

        self._map = mapping

    def __getitem__(self, key: str) -> str:
        return self._map[key]


# ------------------------------------------------------------------ #
# 2. CSV loading & filtering
# ------------------------------------------------------------------ #
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-16")


def parse_filter_string(rule: str) -> Tuple[str, List[str]]:
    if "=" not in rule:
        raise ValueError("Filter must be COLUMN=value1|value2")
    col, val_part = rule.split("=", 1)
    values = [v.strip() for v in val_part.split("|") if v.strip()]
    if not values:
        raise ValueError("Filter needs at least one value")
    return col.strip(), values


def apply_filters(df: pd.DataFrame, rules: List[Tuple[str, List[str]]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, allowed in rules:
        mask &= df[col].astype(str).isin(allowed)
    return df[mask]


# ------------------------------------------------------------------ #
# 3. Metric + stats
# ------------------------------------------------------------------ #
def compute_binary_metric(
    df: pd.DataFrame, group_col: str, metric_col: str, success_vals: List[str]
) -> Dict[str, Dict[str, int]]:
    is_success = df[metric_col].astype(str).isin(success_vals)
    grp = df.groupby(group_col)
    return (
        pd.DataFrame(
            {
                "total": grp.size(),
                "success": grp.apply(lambda g: is_success.loc[g.index].sum()),
            }
        ).to_dict("index")
    )


def two_prop_z(s1: int, n1: int, s2: int, n2: int) -> Tuple[float, float]:
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se2 = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if se2 == 0:
        return 0.0, 1.0
    z = (p1 - p2) / math.sqrt(se2)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def diff_ci(s1: int, n1: int, s2: int, n2: int, alpha: float) -> Tuple[float, float]:
    from scipy.stats import norm

    p1, p2 = s1 / n1, s2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = norm.ppf(1 - alpha / 2)
    return diff - zcrit * se, diff + zcrit * se


def narrative(
    group_a: str,
    group_b: str,
    m1: Dict[str, int],
    m2: Dict[str, int],
    z: float,
    p: float,
    alpha: float,
    ci: Tuple[float, float] | None,
    metric_desc: str = "metric",
) -> str:
    p1 = m1["success"] / m1["total"]
    p2 = m2["success"] / m2["total"]
    diff_pc = (p1 - p2) * 100
    direction = "more" if diff_pc > 0 else "fewer"
    diff_abs = abs(diff_pc)
    sig = "statistically **significant**" if p < alpha else "not statistically significant"

    txt = (
        f"{group_a} recorded {diff_abs:.1f} percentage‑points {direction} {metric_desc} "
        f"than {group_b}. This gap is {sig} (z = {z:.2f}, p = {p:.3f})."
    )
    if ci is not None:
        lo, hi = [x * 100 for x in ci]
        txt += f"  {100*(1-alpha):.0f}% CI: [{lo:+.1f} pp, {hi:+.1f} pp]."
    if min(m1['total'], m2['total']) < 30:
        txt += " ⚠️ Small samples; interpret cautiously."
    return txt


# ------------------------------------------------------------------ #
# 4. Config + CLI
# ------------------------------------------------------------------ #
def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            logging.error("YAML config requested but pyyaml not installed.")
            sys.exit(1)
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic two‑proportion z‑test.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--config", type=Path, help="YAML/JSON config file")

    p.add_argument("--group-column")
    p.add_argument("--groups")
    p.add_argument("--metric-column")
    p.add_argument("--metric-success")
    p.add_argument("--filter", action="append", default=[])

    p.add_argument("--alpha", type=float)
    p.add_argument("--ci", action="store_true")
    p.add_argument("--map", type=Path)
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: List[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    def cfg_or_flag(key: str, default=None):
        return getattr(args, key) if getattr(args, key) not in (None, [], "") else cfg.get(key, default)

    mapper = ColumnMapper(args.map)

    group_col = cfg_or_flag("group_column", mapper["team"])
    metric_col = cfg_or_flag("metric_column", mapper["half"])
    success_vals = [v.strip() for v in cfg_or_flag("metric_success", "Opposition Half").split(",")]

    groups = [g.strip() for g in cfg_or_flag("groups", "").split(",") if g.strip()]
    if len(groups) != 2:
        logging.error("Need exactly two group values via --groups or config.")
        sys.exit(1)
    group_a, group_b = groups

    alpha = float(cfg_or_flag("alpha", 0.05))
    want_ci = bool(cfg_or_flag("ci", False))

    df_raw = load_csv(args.csv)
    df = apply_filters(df_raw, [parse_filter_string(f) for f in args.filter])

    metrics = compute_binary_metric(df, group_col, metric_col, success_vals)
    if {group_a, group_b} - metrics.keys():
        logging.error("One or both groups not found after filtering.")
        sys.exit(1)

    m1, m2 = metrics[group_a], metrics[group_b]
    z, p = two_prop_z(m1["success"], m1["total"], m2["success"], m2["total"])
    ci = diff_ci(m1["success"], m1["total"], m2["success"], m2["total"], alpha) if want_ci else None

    print(narrative(group_a, group_b, m1, m2, z, p, alpha, ci, metric_desc=metric_col))


if __name__ == "__main__":
    main()
