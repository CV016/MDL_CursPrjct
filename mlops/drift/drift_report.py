"""
mlops/drift/drift_report.py
============================
Evidently AI data and model drift monitoring script.

This script is designed to run on a schedule (e.g. daily via cron or a CI
pipeline step) to compare recent inference inputs/outputs against a reference
dataset and produce an HTML drift report.

Workflow:
  1. Pull reference dataset and current-window dataset from PostgreSQL.
  2. Compute text-level features (token count, text length, vocabulary richness).
  3. Run Evidently DataDriftPreset and TextOverviewPreset.
  4. Save the HTML report to the reports/ directory.
  5. Optionally upload the report to the MLflow artifact store.

Usage:
    python mlops/drift/drift_report.py \
        --days-reference 30 \
        --days-current 1 \
        --output-dir mlops/drift/reports
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import psycopg2
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://aidoc:aidoc_secret@localhost:5432/aidoc",
)
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_experiment_logs(
    days_back: int,
    cutoff: datetime,
) -> pd.DataFrame:
    """
    Load experiment_log rows within the window [cutoff - days_back, cutoff)
    from PostgreSQL and derive text features suitable for drift detection.

    Returns a DataFrame with columns:
        variant, input_tokens, latency_ms, created_at
    """
    start_dt = cutoff - timedelta(days=days_back)

    query = """
        SELECT
            variant,
            COALESCE(input_tokens, 0)  AS input_tokens,
            COALESCE(latency_ms, 0.0)  AS latency_ms,
            created_at
        FROM experiment_logs
        WHERE created_at >= %(start_dt)s
          AND created_at <  %(cutoff)s
        ORDER BY created_at
    """

    with psycopg2.connect(DATABASE_URL) as conn:
        df = pd.read_sql(query, conn, params={"start_dt": start_dt, "cutoff": cutoff})

    return df


def _load_feedback(
    days_back: int,
    cutoff: datetime,
) -> pd.DataFrame:
    """Load feedback scores within the specified window."""
    start_dt = cutoff - timedelta(days=days_back)
    query = """
        SELECT variant, score, created_at
        FROM feedback
        WHERE created_at >= %(start_dt)s
          AND created_at <  %(cutoff)s
        ORDER BY created_at
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        df = pd.read_sql(query, conn, params={"start_dt": start_dt, "cutoff": cutoff})
    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Run Evidently drift analysis and write an HTML report.

    Uses:
      - DataDriftPreset   : detects distribution shifts in numerical columns.
      - TextOverviewPreset: not applicable for tabular data — omitted here;
                            extend with TextOverviewPreset if raw text is stored.
    """
    if reference_df.empty or current_df.empty:
        print("[drift] One or both windows have no data — skipping report generation.")
        return

    # Evidently expects at least one prediction or target column; we use
    # `variant` as a categorical prediction proxy and `input_tokens` as a
    # numerical feature.
    column_mapping = ColumnMapping(
        prediction="variant",
        numerical_features=["input_tokens", "latency_ms"],
        categorical_features=["variant"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"[drift] Report saved to {output_path}")


def upload_report_to_mlflow(report_path: Path, run_name: str) -> None:
    """Upload the HTML drift report as an MLflow artifact."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("drift_monitoring")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(str(report_path), artifact_path="drift_reports")
    print(f"[drift] Report uploaded to MLflow run '{run_name}'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Evidently drift report from PostgreSQL experiment logs.",
    )
    parser.add_argument(
        "--days-reference",
        type=int,
        default=30,
        help="Number of days in the reference window (default: 30).",
    )
    parser.add_argument(
        "--days-current",
        type=int,
        default=1,
        help="Number of days in the current window (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mlops/drift/reports"),
        help="Directory where the HTML report is saved.",
    )
    parser.add_argument(
        "--upload-mlflow",
        action="store_true",
        default=False,
        help="Upload the report to the MLflow artifact store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(tz=timezone.utc)

    print(f"[drift] Loading reference window: {args.days_reference} days before {now.date()}")
    reference_df = _load_experiment_logs(args.days_reference, now)

    print(f"[drift] Loading current window: {args.days_current} day(s)")
    current_df = _load_experiment_logs(args.days_current, now)

    print(f"[drift] Reference rows: {len(reference_df)}, Current rows: {len(current_df)}")

    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"drift_report_{timestamp_str}.html"

    generate_drift_report(reference_df, current_df, report_path)

    if args.upload_mlflow and report_path.exists():
        upload_report_to_mlflow(report_path, run_name=f"drift_{timestamp_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
