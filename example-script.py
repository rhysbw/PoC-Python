#!/usr/bin/env python3
"""
PoC script for SPORTSTATS Lite: import a Sportscode-exported CSV of pass events,
compute simple two-proportion z-test comparing proportion of passes in the opponent’s half,
and print a user-friendly interpretation.

Future extensions:
 - More statistical tests (chi-square, logistic regression, lag-sequential analysis)
 - Reading from varied export formats (XML, JSON)
 - Minimal GUI or CLI prompts for choosing metrics
 - Integration with Sportscode API or export automation
"""

import csv
import math
import sys


def read_event_data(csv_path):
    """
    Reads event data from a Sportscode-exported CSV file.

    Expects columns including:
      - 'Row': the team or entity performing the event
      - 'Half': 'Opposition Half' or 'Own Half'
      - 'Pass Type': e.g., 'Regular Pass', 'Kick Off', etc.
      - 'Outcome': 'Complete' or 'Incomplete'

    Returns:
        List[dict]: each dict corresponds to one event row.
    """
    events = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)

    return events


def compute_pass_metrics(events, teams=None, pass_types=None):
    """
    Computes pass counts for each team:
      - total_passes: number of events matching pass_types (default all pass types)
      - passes_in_opponent_half: count of those that occurred in 'Opposition Half'

    Args:
        events (List[dict]): output of read_event_data()
        teams (iterable[str], optional): which teams to include (default: all found)
        pass_types (iterable[str], optional): which pass types to count (default: all 'Pass')

    Returns:
        Dict[str, Dict[str,int]]: mapping each team to its pass metrics.
    """
    metrics = {}
    # Determine teams
    all_teams = set(e['Row'] for e in events)
    if teams is None:
        teams = all_teams

    # Determine pass types
    if pass_types is None:
        # include any type containing 'Pass'
        pass_types = None

    for event in events:
        team = event['Row']
        if team not in teams:
            continue
        ptype = event.get('Pass Type', '')
        """if pass_types is None:
            if 'Pass' not in ptype:
                continue
        else:
            if ptype not in pass_types:
                continue"""

        # initialize counters
        if team not in metrics:
            metrics[team] = {'total_passes': 0, 'passes_in_opponent_half': 0}

        metrics[team]['total_passes'] += 1
        if event.get('Half', '') == 'Opposition Half':
            metrics[team]['passes_in_opponent_half'] += 1

    return metrics


def two_proportion_z_test(success1, n1, success2, n2):
    """
    Performs a two-proportion z-test, with a guard against zero variance.

    Returns:
        z_stat (float), p_value (float)
    """
    p1 = success1 / n1
    p2 = success2 / n2
    pooled = (success1 + success2) / (n1 + n2)
    se_term = pooled * (1 - pooled) * (1/n1 + 1/n2)
    
    # If no variance (e.g. both success counts 0 or both full), return no effect
    if se_term == 0:
        print("No variance, no effect. Got 0.")
        return 0.0, 1.0
    
    se = math.sqrt(se_term)
    z = (p1 - p2) / se
    # two-tailed p-value
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - cdf)
    return z, p_value


def interpret_result(z_stat, p_value, alpha=0.05):
    """
    Returns a plain-English interpretation of the test result.
    """
    if p_value < alpha:
        return f"Difference is statistically significant (z = {z_stat:.2f}, p = {p_value:.3f} < {alpha})"
    else:
        return f"No significant difference (z = {z_stat:.2f}, p = {p_value:.3f} >= {alpha})"


def main():
    if len(sys.argv) < 2:
        print("Usage: python poc_sportstats.py <events_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    events = read_event_data(csv_path)

    # Compute metrics for all teams
    metrics = compute_pass_metrics(events)

    # Expect exactly two teams for comparison
    teams = list(metrics.keys())
    if len(teams) != 2:
        print(f"Expected data for 2 teams, found {len(teams)}. Metrics: {metrics}")
        sys.exit(1)

    t1, t2 = teams
    m1 = metrics[t1]
    m2 = metrics[t2]

    # Run statistical test
    z_stat, p_val = two_proportion_z_test(
        m1['passes_in_opponent_half'], m1['total_passes'],
        m2['passes_in_opponent_half'], m2['total_passes']
    )

    # Output
    print(f"Comparison: {t1} vs {t2}")
    print(f"Metrics: {t1}: {m1}, {t2}: {m2}")
    print(interpret_result(z_stat, p_val))


if __name__ == '__main__':
    main()
