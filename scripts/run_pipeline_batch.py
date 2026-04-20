#!/usr/bin/env python3
"""
End-to-end batch run without the Streamlit UI:

  1. For each CSV row, POST /process (same as run_traffic_simulator.py).
  2. The backend already runs drift detection, summarisation, questions, MLflow,
     Prometheus metrics, and (if JUDGE_ENABLED) the automated NLI judge in
     BackgroundTasks after the response is returned — there is no separate
     judge endpoint to call.
  3. This script records each HTTP response to a JSONL file and optionally
     polls MLflow until automated_reward_score appears for that run_id.

Run on the server with --backend http://localhost:8000 and
--mlflow-uri http://localhost:5000 (host ports mapped from compose).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import pandas as pd
import requests
from tqdm import tqdm


def _poll_automated_metrics(run_id: str, tracking_uri: str, max_wait_s: float) -> Dict[str, Any]:
    """Block until MLflow shows judge metrics for run_id or timeout."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {"poll_error": "install mlflow: pip install mlflow"}

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            run = client.get_run(run_id)
            m = run.data.metrics
            if "automated_reward_score" in m or "automated_is_thumbs_up" in m:
                return {
                    "automated_reward_score": m.get("automated_reward_score"),
                    "automated_is_thumbs_up": m.get("automated_is_thumbs_up"),
                }
        except Exception as exc:
            return {"poll_error": str(exc)}
        time.sleep(0.4)
    return {"poll_error": "timeout waiting for automated judge metrics (is JUDGE_ENABLED?)"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POST CSV rows to /process and record JSONL + optional MLflow judge poll.",
    )
    parser.add_argument("--csv", default="test_traffic.csv")
    parser.add_argument(
        "--backend",
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MOCK_JWT_TOKEN", "mock-jwt-token-for-academic-project"),
    )
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--output",
        default="pipeline_results.jsonl",
        help="Append one JSON object per row (newline-delimited).",
    )
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Include full summary text in JSONL (large files).",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="If set, poll each run for automated judge metrics (e.g. http://localhost:5000).",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for automated_reward_score per run.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        print("CSV must contain a 'text' column.", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {args.token}"}
    url = f"{args.backend.rstrip('/')}/process"
    errors = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="pipeline batch"):
            text = str(row["text"])
            order = row["upload_order"] if "upload_order" in df.columns else idx + 1
            name = f"traffic_{int(order)}.txt"
            files = {"file": (name, text.encode("utf-8"), "text/plain; charset=utf-8")}
            record = {
                "upload_order": int(order),
                "ok": False,
            }
            try:
                resp = requests.post(url, files=files, headers=headers, timeout=args.timeout)
                record["http_status"] = resp.status_code
                if resp.status_code >= 400:
                    errors += 1
                    record["error"] = resp.text[:2000]
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    continue
                data = resp.json()
                record["ok"] = True
                record["run_id"] = data.get("run_id")
                record["model_name"] = data.get("model_name")
                record["inference_latency"] = data.get("inference_latency")
                record["drift_status"] = data.get("drift_status")
                record["chunk_count"] = data.get("chunk_count")
                record["num_questions"] = len(data.get("questions") or [])
                if args.include_summary:
                    record["summary"] = data.get("summary")
                    record["questions"] = data.get("questions")
                else:
                    record["summary_chars"] = len(data.get("summary") or "")

                if args.mlflow_uri and record.get("run_id"):
                    record["mlflow"] = _poll_automated_metrics(
                        record["run_id"],
                        args.mlflow_uri.rstrip("/"),
                        args.poll_timeout,
                    )
            except requests.RequestException as exc:
                errors += 1
                record["error"] = str(exc)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            time.sleep(args.delay)

    if errors:
        print(f"Completed with {errors} error row(s). See {args.output}.", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
