import io
import pandas as pd
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the functions directly from the script (adjust if renamed to a package)
from poc_sportstats import (
    apply_filters,
    compute_metrics,
    two_proportion_z,
)

# --- Helpers -----------------------------------------------------------
SAMPLE_CSV = """\
Timeline,Start time,Duration,Row,Instance num,Pass Type,Player,Recipient,Half,Outcome
M1,00.0,1.0,TeamA,1,Regular Pass,PA,PB,Opposition Half,Complete
M1,00.1,1.0,TeamA,2,Regular Pass,PA,PB,Own Half,Complete
M1,00.2,1.0,TeamB,1,Regular Pass,PB,PA,Opposition Half,Complete
M1,00.3,1.0,TeamB,2,Regular Pass,PB,PA,Opposition Half,Incomplete
"""

def sample_df():
    return pd.read_csv(io.StringIO(SAMPLE_CSV))


# --- Tests -------------------------------------------------------------
def test_apply_filters_completed_only():
    df = sample_df()
    filtered = apply_filters(df, completed_only=True, pass_types=[])
    # One TeamB row should be dropped (Outcome == Incomplete)
    assert len(filtered) == 3
    assert filtered["Outcome"].eq("Incomplete").sum() == 0


def test_compute_metrics_counts():
    df = sample_df()
    metrics = compute_metrics(df)
    assert metrics["TeamA"]["total_passes"] == 2
    assert metrics["TeamA"]["passes_in_opponent_half"] == 1
    assert metrics["TeamB"]["total_passes"] == 2
    assert metrics["TeamB"]["passes_in_opponent_half"] == 2


def test_two_proportion_z_result():
    # TeamA: 1/2, TeamB: 2/2  → diff = -0.5
    z, p = two_proportion_z(1, 2, 2, 2)
    # Direction doesn’t matter for two‑tailed p; check magnitude & p < 0.1
    assert round(z, 2) == -1.41
    assert p < 0.16
