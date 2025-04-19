#!/usr/bin/env python3
"""
SPORTSTATS-Lite - Logic + CLI  (v0.4)
-------------------------------------
• Two-proportion z-test  (--test prop)
• Lag-sequential χ² test (--test lag)
• YAML/JSON config, column mapping, filters
"""

from __future__ import annotations
import argparse, json, logging, math, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm  # NEW import for lag test & CI

try:
    import yaml  # optional
except ModuleNotFoundError:
    yaml = None


# ---------- 1. Column mapping ------------------------------------------
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
                txt = mapping_path.read_text()
                if mapping_path.suffix.lower() in {".yml", ".yaml"}:
                    if yaml is None:
                        raise RuntimeError("pyyaml not installed.")
                    mapping.update(yaml.safe_load(txt))
                else:
                    mapping.update(json.loads(txt))
            except Exception as exc:
                logging.error("Failed to read mapping %s: %s", mapping_path, exc)
                sys.exit(1)
        if missing := self.KEYS - mapping.keys():
            logging.error("Mapping file missing keys: %s", ", ".join(sorted(missing)))
            sys.exit(1)
        self._map = mapping

    def __getitem__(self, key: str) -> str:
        return self._map[key]


# ---------- 2. CSV + filters -------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-16")


def parse_filter_string(rule: str) -> Tuple[str, List[str]]:
    if "=" not in rule:
        raise ValueError("Filter must be COLUMN=value1|value2")
    col, vals = rule.split("=", 1)
    return col.strip(), [v.strip() for v in vals.split("|") if v.strip()]


def apply_filters(df: pd.DataFrame, rules: List[Tuple[str, List[str]]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, allowed in rules:
        mask &= df[col].astype(str).isin(allowed)
    return df[mask]


# ---------- 3. Metrics & stats -----------------------------------------
def compute_binary_metric(
    df: pd.DataFrame, group_col: str, metric_col: str, success_vals: List[str]
) -> Dict[str, Dict[str, int]]:
    ok = df[metric_col].astype(str).isin(success_vals)
    grp = df.groupby(group_col)
    return (
        pd.DataFrame({"total": grp.size(), "success": grp.apply(lambda g: ok[g.index].sum())})
        .to_dict("index")
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
    p1, p2 = s1 / n1, s2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = norm.ppf(1 - alpha / 2)
    return diff - zcrit * se, diff + zcrit * se


def compute_lag_sequential(
    df: pd.DataFrame, group_col: str, metric_col: str, success_vals: List[str]
) -> Dict[str, Dict[str, float]]:
    """χ² & Yule Q per group (lag 1)."""
    res: Dict[str, Dict[str, float]] = {}
    for g, sub in df.groupby(group_col):
        seq = sub[metric_col].astype(str).isin(success_vals).to_numpy(dtype=int)
        if len(seq) < 2:
            res[g] = dict(n11=0, n10=0, n01=0, n00=0, chi2=0, p=1, yule_q=0)
            continue
        prev, curr = seq[:-1], seq[1:]
        n11 = int(((prev == 1) & (curr == 1)).sum())
        n10 = int(((prev == 1) & (curr == 0)).sum())
        n01 = int(((prev == 0) & (curr == 1)).sum())
        n00 = int(((prev == 0) & (curr == 0)).sum())
        chi2, p, _, _ = chi2_contingency([[n11, n10], [n01, n00]], correction=False)
        denom = n11 * n00 + n10 * n01
        yq = ((n11 * n00) - (n10 * n01)) / denom if denom else 0.0
        res[g] = dict(n11=n11, n10=n10, n01=n01, n00=n00, chi2=chi2, p=p, yule_q=yq)
    return res


def prop_narrative(
    group_a: str,
    group_b: str,
    m1: Dict[str, int],
    m2: Dict[str, int],
    z: float,
    p: float,
    alpha: float,
    ci: Tuple[float, float] | None,
    metric_desc: str,
    hypothesis_txt: str | None = None,
) -> str:
    p1, p2 = m1["success"] / m1["total"], m2["success"] / m2["total"]
    diff_pct = (p1 - p2) * 100
    decision = "Reject H₀" if p < alpha else "Fail to reject H₀"
    lines = []

    if hypothesis_txt:
        lines.append(f"**Hypothesis:** {hypothesis_txt}")

    direction = "more" if diff_pct > 0 else "fewer"
    sig = "statistically **significant**" if p < alpha else "not statistically significant"
    
    

    lines.append(f"*Null H₀:* the proportion of {metric_desc} is equal in both groups.")
    lines.append(f"**Decision:** {decision} (z = {z:.2f}, p = {p:.3f}, α = {alpha})")
    lines.append(
        f"Observed gap: {abs(diff_pct):.1f} pp "
        f"({'higher' if diff_pct>0 else 'lower'} for {group_a})."
    )

    lines.append(f"\n{group_a} recorded {abs(diff_pct):.1f} pp {direction} {metric_desc} than {group_b}. ")
    if ci:
        lo, hi = [x * 100 for x in ci]
        lines.append(f"{100*(1-alpha):.0f}% CI for gap: [{lo:+.1f}, {hi:+.1f}] pp.")
    return "\n".join(lines)

# Constants for effect-size thresholds
Q_MODERATE = 0.3
Q_STRONG = 0.5

def lag_narrative(
    team: str,
    stats: Dict[str, float],
    alpha: float = 0.05,
    metric_desc: str | None = None,
    df: int = 1,
    q_thresholds: tuple[float, float] = (Q_MODERATE, Q_STRONG)
) -> str:
    """
    Generate a Markdown narrative for a lag-sequential test result.

    Parameters:
    - team: Name of the team or event
    - stats: dict with keys "chi2", "p", "yule_q"
    - alpha: significance level for the χ² test
    - metric_desc: human-readable name of the event being tested
    - df: degrees of freedom for the χ² statistic
    - q_thresholds: (moderate, strong) effect-size breakpoints for Yule's Q

    Returns:
    - A formatted Markdown string.
    """
    chi2, p, q = stats["chi2"], stats["p"], stats["yule_q"]
    desc = metric_desc or "the event"

    # Decision based on p-value
    reject = p < alpha
    decision = "Reject H₀" if reject else "Fail to reject H₀"

    # Format p-value neatly
    if p < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"= {p:.3f}"

    # Only describe a trend if the test is significant
    if not reject:
        trend = "independent (no clear carry-over)"
    else:
        mod, strong = q_thresholds
        if abs(q) >= strong:
            strength = "strongly "
        elif abs(q) >= mod:
            strength = ""
        else:
            strength = "weakly "

        if q > 0:
            trend = f"{strength}persistent (event tends to repeat)"
        else:
            trend = f"{strength}alternating (event tends to switch away)"

    return (
        f"**{team}** - χ²({df}) = {chi2:.2f}, p {p_str}, Yule's Q = {q:.2f}\n"
        f"*Null H₀:* occurrence of {desc} is independent of the previous event.\n"
        f"**Decision:** {decision}. Pattern appears **{trend}**."
    )

def suggest_hypotheses(
    df: pd.DataFrame,
    *,
    max_groups: int = 8,
    max_suggestions: int = 5,
    min_rows_per_group: int = 20,
    include_lag: bool = True,
) -> list[dict]:
    """
    Return up to *max_suggestions* ranked hypothesis dictionaries, skipping
    trivial or self-referential pairs.
    """
    ideas: list[dict] = []

    # 1. Candidate columns -------------------------------------------------
    group_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and 2 <= df[c].nunique(dropna=True) <= max_groups
    ]

    binary_cols: list[str] = []
    for c in df.columns:
        nunq = df[c].nunique(dropna=True)
        if nunq == 2:
            binary_cols.append(c)
        elif nunq <= 4 and df[c].dtype == "object":
            counts = df[c].value_counts(dropna=True)
            if counts.iloc[0] / counts.sum() > 0.8:
                binary_cols.append(c)

    # 2. Pair evaluation ---------------------------------------------------
    for gcol in group_cols:
        grp_counts = df[gcol].value_counts(dropna=True)
        if len(grp_counts) < 2:
            continue
        g1, g2 = grp_counts.index[:2]
        n1, n2 = int(grp_counts[g1]), int(grp_counts[g2])
        if min(n1, n2) < min_rows_per_group:
            continue

        for mcol in binary_cols:
            # --- guard 0: skip comparing a column with itself -------------
            if mcol == gcol:
                continue

            vals = df[mcol].dropna().unique().astype(str)
            succ = [str(vals[0])]  # use first value as 'success'

            # --- guard 1: skip if success is identical to group names -----
            if succ[0] in (g1, g2):
                continue

            # ---------- proportion idea -----------------------------------
            p1 = (df[df[gcol] == g1][mcol].astype(str).isin(succ)).mean()
            p2 = (df[df[gcol] == g2][mcol].astype(str).isin(succ)).mean()
            gap = abs(p1 - p2)

            # --- guard 2: skip near-zero or near-100% gaps ----------------
            if gap < 0.05 or gap > 0.95:
                continue

            prop_score = gap * np.sqrt(min(n1, n2))
            ideas.append(
                dict(
                    test="prop",
                    group_column=gcol,
                    groups=[g1, g2],
                    metric_column=mcol,
                    success_vals=succ,
                    score=prop_score,
                    readable=(
                        f"Compare how often **{g1}** vs **{g2}** record “{succ[0]}” in "
                        f"**{mcol}** (current gap ≈ {gap*100:.1f} pp)."
                    ),
                )
            )

            # ---------- lag idea ------------------------------------------
            if include_lag:
                sub = df[df[gcol].isin([g1, g2])].sort_index()
                seq = sub[mcol].astype(str).isin(succ).to_numpy(dtype=int)
                if len(seq) < 3:
                    continue
                prev, curr = seq[:-1], seq[1:]
                n11 = ((prev == 1) & (curr == 1)).sum()
                n10 = ((prev == 1) & (curr == 0)).sum()
                n01 = ((prev == 0) & (curr == 1)).sum()
                n00 = ((prev == 0) & (curr == 0)).sum()
                den = n11 * n00 + n10 * n01
                if den == 0:
                    continue
                yq = (n11 * n00 - n10 * n01) / den
                lag_score = abs(yq) * np.sqrt(n11 + n10 + n01 + n00 - 2)

                # guard 2 reused: ignore trivial zero Q
                if abs(yq) < 0.05:
                    continue

                ideas.append(
                    dict(
                        test="lag",
                        group_column=gcol,
                        groups=[g1, g2],
                        metric_column=mcol,
                        success_vals=succ,
                        score=lag_score,
                        readable=(
                            f"See if “{succ[0]}” in **{mcol}** tends to repeat "
                            f"back-to-back within **{g1}** or **{g2}** possessions."
                        ),
                    )
                )

    # 3. Rank & return -----------------------------------------------------
    ideas.sort(key=lambda d: d["score"], reverse=True)
    return ideas[:max_suggestions]


# ---------- 4. Config + CLI --------------------------------------------
def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    txt = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            logging.error("pyyaml required for YAML config.")
            sys.exit(1)
        return yaml.safe_load(txt)
    return json.loads(txt)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-proportion or Lag-sequential test.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--test", choices=["prop", "lag"], default="prop")
    p.add_argument("--group-column"), p.add_argument("--groups")
    p.add_argument("--metric-column"), p.add_argument("--metric-success")
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
    get = lambda k, d=None: (getattr(args, k) or cfg.get(k, d))

    mapper = ColumnMapper(args.map)
    group_col = get("group_column", mapper["team"])
    metric_col = get("metric_column", mapper["half"])
    success_vals = [v.strip() for v in get("metric_success", "Opposition Half").split(",")]

    groups_str = get("groups")
    if not groups_str or "," not in groups_str:
        logging.error("Provide two comma-separated values via --groups or config.")
        sys.exit(1)
    group_a, group_b = [g.strip() for g in groups_str.split(",", 1)]
    alpha = float(get("alpha", 0.05))

    df = apply_filters(load_csv(args.csv), [parse_filter_string(f) for f in args.filter])

    if args.test == "prop":
        met = compute_binary_metric(df, group_col, metric_col, success_vals)
        z, p = two_prop_z(met[group_a]["success"], met[group_a]["total"],
                          met[group_b]["success"], met[group_b]["total"])
        ci = diff_ci(met[group_a]["success"], met[group_a]["total"],
                     met[group_b]["success"], met[group_b]["total"], alpha) if args.ci else None
        print(prop_narrative(group_a, group_b, met[group_a], met[group_b], z, p, alpha, ci,
                        metric_desc=f"{metric_col}∈{success_vals}"))
    elif args.test == "lag":
        stats = compute_lag_sequential(df, group_col, metric_col, success_vals)
        print(lag_narrative(group_a, stats[group_a], alpha,
                            metric_desc=f"{metric_col} ∈ {success_vals}"))
        print(lag_narrative(group_b, stats[group_b], alpha,
                            metric_desc=f"{metric_col} ∈ {success_vals}"))



if __name__ == "__main__":
    main()
